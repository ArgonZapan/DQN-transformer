"""
TelegramCommands — long-polling bot receiving commands from the configured chat.

Available commands:
  /status     — quick overview: step, sps, epsilon, loss, buffer
  /health     — immediate health check
  /pause      — pause the training loop
  /resume     — resume the training loop
  /checkpoint — force checkpoint save
  /buffer     — buffer statistics
  /actors     — actor status
  /stop       — graceful shutdown
  /restart    — reset model, buffer and checkpoints (fresh training)
  /delete     — delete checkpoints and ONNX model (requires confirmation: /delete confirm)
  /raport     — force immediate training report
  /set        — set config value: /set section.key value
  /get        — get config value: /get section[.key]
  /ereset     — temporary exploration reset: /ereset [target] [duration_steps]

Bot uses long-polling every 2s. Token and chat_id taken from [report] or [alerts] in config.toml.
"""

import glob
import json
import logging
import os
import subprocess
import threading
import time
import urllib.parse
import urllib.request

logger = logging.getLogger('learner')

# Katalog główny projektu — dwa poziomy wyżej od python/diagnostics/
_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

_TG_LIMIT = 4096


def _split_message(text: str) -> list[str]:
    if len(text) <= _TG_LIMIT:
        return [text]
    chunks, current, size = [], [], 0
    for line in text.split('\n'):
        line_len = len(line) + 1
        if size + line_len > _TG_LIMIT and current:
            chunks.append('\n'.join(current))
            current, size = [], 0
        current.append(line)
        size += line_len
    if current:
        chunks.append('\n'.join(current))
    return chunks


class TelegramCommands:
    def __init__(self, config, trainer, training_report):
        alerts_cfg = config.get('alerts', {})
        report_cfg = config.get('report', {})

        self._token   = report_cfg.get('telegram_token',   '') or alerts_cfg.get('telegram_token',   '')
        self._chat_id = str(report_cfg.get('telegram_chat_id', '') or alerts_cfg.get('telegram_chat_id', ''))
        self._trainer = trainer
        self._report  = training_report

        self._last_update_id      = 0
        self._pending_delete      = False  # oczekiwanie na /delete confirm

        self.enabled = bool(self._token and self._chat_id)
        if not self.enabled:
            logger.warning('[Commands] Brak telegram_token lub chat_id — komendy wyłączone')

    # ── publiczne API ─────────────────────────────────────────────────────────

    def start(self):
        """Uruchamia wątek pollingu w tle. Nieblokujące."""
        if not self.enabled:
            return
        threading.Thread(target=self._poll_loop, daemon=True, name='TelegramCommands').start()
        logger.info('[Commands] Telegram command polling started')

    # ── polling ───────────────────────────────────────────────────────────────

    def _poll_loop(self):
        while True:
            try:
                updates = self._get_updates()
                for upd in updates:
                    try:
                        self._handle(upd)
                    except Exception as e:
                        logger.warning(f'[Commands] Handle error: {e}')
            except Exception as e:
                logger.debug(f'[Commands] Poll error: {e}')
            time.sleep(2)

    def _get_updates(self) -> list:
        url = (
            f'https://api.telegram.org/bot{self._token}/getUpdates'
            f'?offset={self._last_update_id + 1}&timeout=0'
        )
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data.get('result', [])

    # ── dispatch ──────────────────────────────────────────────────────────────

    def _handle(self, update: dict):
        self._last_update_id = update['update_id']
        msg     = update.get('message', {})
        chat_id = str(msg.get('chat', {}).get('id', ''))

        # Ignoruj wiadomości spoza skonfigurowanego chatu
        if chat_id != self._chat_id:
            return

        text = msg.get('text', '').strip()
        cmd  = text.lower().split('@')[0]  # usuń @BotName suffix

        if cmd == '/restart':
            self._cmd_restart()
        elif cmd == '/delete' and text.lower() == '/delete confirm':
            self._cmd_delete_confirm()
        elif cmd == '/delete':
            self._cmd_delete_prompt()
        elif cmd in ('/raport', '/report'):
            self._cmd_raport()
        elif text.lower().startswith('/set '):
            self._cmd_set(text[5:].strip())
        elif cmd == '/set':
            self._send('Użycie: <code>/set sekcja.klucz wartość</code>\nPrzykład: <code>/set reward.loss_scale 1.5</code>')
        elif text.lower().startswith('/get '):
            self._cmd_get(text[5:].strip())
        elif cmd == '/get':
            self._cmd_get('')
        elif cmd in ('/ereset', '/epsilon_reset') or text.lower().startswith('/ereset '):
            self._cmd_ereset(text[len('/ereset'):].strip() if text.lower().startswith('/ereset') else '')
        elif cmd == '/status':
            self._cmd_status()
        elif cmd == '/health':
            self._cmd_health()
        elif cmd == '/pause':
            self._cmd_pause()
        elif cmd == '/resume':
            self._cmd_resume()
        elif cmd == '/checkpoint':
            self._cmd_checkpoint()
        elif cmd == '/buffer':
            self._cmd_buffer()
        elif cmd == '/actors':
            self._cmd_actors()
        elif cmd == '/stop':
            self._cmd_stop()
        elif cmd.startswith('/'):
            self._send('Nieznana komenda.\nDostępne: /status | /health | /pause | /resume | /checkpoint | /buffer | /actors | /stop | /restart | /delete | /raport | /set | /get | /ereset')

    # ── komendy ───────────────────────────────────────────────────────────────

    def _cmd_status(self):
        import time as _time
        t = self._trainer
        elapsed = _time.time() - t._sps_time_anchor
        steps_delta = t.step_count - t._sps_step_anchor
        sps = steps_delta / elapsed if elapsed > 0 else 0.0
        buf_size  = len(t.buffer)
        buf_cap   = t.buffer.capacity if hasattr(t.buffer, 'capacity') else t.config['training']['buffer_capacity']
        buf_pct   = 100.0 * buf_size / buf_cap if buf_cap else 0.0
        paused    = '⏸ PAUZA' if getattr(t, '_paused', False) else '▶ Trenuje'
        self._send(
            f'📊 <b>Status</b> [{paused}]\n'
            f'  Krok:    <code>{t.step_count:,}</code>\n'
            f'  SPS:     <code>{sps:.1f}</code>\n'
            f'  ε:       <code>{t.epsilon:.4f}</code>\n'
            f'  Loss:    <code>{t.last_loss:.5f}</code>\n'
            f'  Bufor:   <code>{buf_size:,} / {buf_cap:,} ({buf_pct:.1f}%)</code>'
        )

    def _cmd_health(self):
        self._send('🔍 Uruchamiam health check...')
        import threading
        threading.Thread(target=self._cmd_health_bg, daemon=True, name='HealthBG').start()

    def _cmd_health_bg(self):
        try:
            from diagnostics.health_runner import HealthRunner
            hr = HealthRunner(self._trainer, check_interval_sec=0)
            result = hr._run_check()
            lines = ['🏥 <b>Health Check</b>']
            for k, v in result.items():
                if k == 'step':
                    continue
                if isinstance(v, float):
                    lines.append(f'  <code>{k}</code>: <code>{v:.4f}</code>')
                else:
                    lines.append(f'  <code>{k}</code>: <code>{v}</code>')
            self._send('\n'.join(lines))
        except Exception as e:
            self._send(f'❌ Health check error: {e}')
            logger.warning(f'[Commands] Health check failed: {e}')

    def _cmd_pause(self):
        if getattr(self._trainer, '_paused', False):
            self._send('⚠️ Już w trybie pauzy.')
            return
        self._trainer._paused = True
        self._send('⏸ Pętla treningowa <b>wstrzymana</b>.')
        logger.info('[Commands] Training paused via Telegram')

    def _cmd_resume(self):
        if not getattr(self._trainer, '_paused', False):
            self._send('⚠️ Trening nie jest wstrzymany.')
            return
        self._trainer._paused = False
        self._send('▶ Pętla treningowa <b>wznowiona</b>.')
        logger.info('[Commands] Training resumed via Telegram')

    def _cmd_checkpoint(self):
        try:
            self._trainer.save_checkpoint(include_buffer=False)
            self._send(f'💾 Checkpoint zapisany (step {self._trainer.step_count:,}).')
        except Exception as e:
            self._send(f'❌ Błąd zapisu checkpointu: {e}')
            logger.warning(f'[Commands] Checkpoint save failed: {e}')

    def _cmd_buffer(self):
        t   = self._trainer
        buf = t.buffer
        size = len(buf)
        cap  = buf.capacity if hasattr(buf, 'capacity') else t.config['training']['buffer_capacity']
        fill_pct = 100.0 * size / cap if cap else 0.0
        lines = [
            '📦 <b>Replay Buffer</b>',
            f'  Rozmiar:   <code>{size:,} / {cap:,} ({fill_pct:.1f}%)</code>',
        ]
        if hasattr(buf, '_max_priority'):
            lines.append(f'  Max prior: <code>{buf._max_priority:.4f}</code>')
        cfg_per = t.config.get('per', {})
        lines += [
            f'  α (PER):   <code>{cfg_per.get("alpha", "?")}</code>',
            f'  β (PER):   <code>{cfg_per.get("beta_start", "?")} → {cfg_per.get("beta_end", "?")}</code>',
            f'  pos_ratio: <code>{cfg_per.get("positive_ratio", "?")}</code>',
        ]
        self._send('\n'.join(lines))

    def _cmd_actors(self):
        t = self._trainer
        actor_data = dict(getattr(t, '_actor_metrics', {}))
        if not actor_data:
            self._send('ℹ️ Brak danych od aktorów.')
            return
        lines = ['🎭 <b>Aktorzy</b>']
        for symbol, data in actor_data.items():
            wins   = data.get('wins', 0)
            losses = data.get('losses', 0)
            total  = wins + losses
            wr     = wins / total * 100 if total else 0.0
            actions_total = (data.get('action_hold', 0) + data.get('action_long', 0) +
                             data.get('action_short', 0) + data.get('action_close', 0))
            hold_pct = data.get('action_hold', 0) / actions_total * 100 if actions_total else 0.0
            pnl = data.get('total_pnl', 0.0)
            lines.append(
                f'\n<b>{symbol}</b>\n'
                f'  Transakcje: <code>{total}</code>  WR: <code>{wr:.1f}%</code>\n'
                f'  HOLD: <code>{hold_pct:.1f}%</code>  PnL: <code>{pnl:.4f}</code>'
            )
        self._send('\n'.join(lines))

    def _cmd_stop(self):
        self._send('🛑 Inicjuję graceful shutdown...')
        flag = os.path.abspath(os.path.join(_BASE_DIR, 'python', 'shutdown.flag'))
        try:
            with open(flag, 'w') as f:
                f.write('telegram')
            logger.info('[Commands] Shutdown flag created via Telegram')
        except Exception as e:
            self._send(f'❌ Nie można utworzyć shutdown.flag: {e}')

    def _cmd_restart(self):
        self._send('🔄 Resetuję szkolenie...\nCzyszczę checkpointy, model i bufor.')
        import threading
        threading.Thread(target=self._cmd_restart_bg, daemon=True, name='RestartBG').start()

    def _cmd_restart_bg(self):
        try:
            # 1. Usuń checkpointy z dysku
            ckpt_dir = os.path.join(_BASE_DIR, 'python', 'checkpoints')
            files = (
                glob.glob(os.path.join(ckpt_dir, '*.pt'))  +
                glob.glob(os.path.join(ckpt_dir, '*.pth')) +
                glob.glob(os.path.join(ckpt_dir, '*.onnx'))
            )
            deleted = 0
            for f in files:
                try:
                    os.remove(f)
                    deleted += 1
                except Exception as e:
                    logger.warning(f'[Commands] Delete {f}: {e}')

            # 2. Reset modelu i szkolenia w trainerze
            self._trainer.reset_training()

            self._send(
                f'✅ Reset zakończony\n'
                f'  Usunięto: {deleted} plików\n'
                f'  Model: świeże wagi\n'
                f'  Bufor: wyczyszczony\n'
                f'  Krok: 0, ε={self._trainer.epsilon:.4f}'
            )
            logger.info('[Commands] Training reset completed')
        except Exception as e:
            self._send(f'❌ Błąd resetu: {e}')
            logger.error(f'[Commands] Reset failed: {e}')

    def _cmd_delete_prompt(self):
        ckpt_dir = os.path.join(_BASE_DIR, 'python', 'checkpoints')
        files = (
            glob.glob(os.path.join(ckpt_dir, '*.pt'))  +
            glob.glob(os.path.join(ckpt_dir, '*.pth')) +
            glob.glob(os.path.join(ckpt_dir, '*.onnx'))
        )
        self._pending_delete = True
        self._send(
            f'UWAGA: Usunie <b>{len(files)} plików</b> checkpointów i modeli ONNX.\n'
            f'Wyślij <code>/delete confirm</code> aby potwierdzić.'
        )

    def _cmd_delete_confirm(self):
        if not self._pending_delete:
            self._send('Brak oczekującego potwierdzenia. Wyślij najpierw /delete.')
            return
        self._pending_delete = False

        ckpt_dir = os.path.join(_BASE_DIR, 'python', 'checkpoints')
        files = (
            glob.glob(os.path.join(ckpt_dir, '*.pt'))  +
            glob.glob(os.path.join(ckpt_dir, '*.pth')) +
            glob.glob(os.path.join(ckpt_dir, '*.onnx'))
        )
        deleted, failed = [], []
        for f in files:
            try:
                os.remove(f)
                deleted.append(os.path.basename(f))
            except Exception as e:
                failed.append(os.path.basename(f))
                logger.warning(f'[Commands] Delete failed: {f}: {e}')

        lines = [f'Usunięto <b>{len(deleted)}</b> plików.']
        if deleted:
            lines += [f'  • {n}' for n in deleted]
        if failed:
            lines.append(f'\nNie udało się usunąć {len(failed)}: {", ".join(failed)}')
        self._send('\n'.join(lines))

    def _cmd_ereset(self, args: str):
        """Wywołuje trigger_epsilon_reset z opcjonalnymi parametrami.
        Użycie: /ereset           — wartości z config
                /ereset 0.9       — target=0.9, duration z config
                /ereset 0.9 3000  — target=0.9, duration=3000 kroków
        """
        t = self._trainer
        cfg = t.config['training']
        parts = args.split()
        try:
            target   = float(parts[0]) if len(parts) >= 1 else cfg.get('epsilon_reset_target',   0.8)
            duration = int(parts[1])   if len(parts) >= 2 else cfg.get('epsilon_reset_duration', 5000)
        except (ValueError, IndexError):
            self._send('❌ Zły format. Użycie: <code>/ereset [target] [duration_steps]</code>\nPrzykład: <code>/ereset 0.9 3000</code>')
            return

        if not (0.0 < target <= 1.0):
            self._send(f'❌ target musi być z zakresu (0, 1], podano: {target}')
            return

        t.trigger_epsilon_reset(target=target, duration=duration)
        end_step = t._epsilon_reset_end_step
        self._send(
            f'🔄 <b>Epsilon reset aktywny</b>\n'
            f'  ε → <code>{target:.2f}</code> przez <code>{duration:,}</code> kroków\n'
            f'  Kończy się na kroku: <code>{end_step:,}</code>'
        )

    def _cmd_raport(self):
        self._send('⏳ Generuję raport + backtest...')
        import threading
        threading.Thread(target=self._cmd_raport_bg, daemon=True, name='RaportBG').start()

    def _cmd_raport_bg(self):
        try:
            from diagnostics.backtest_runner import run_backtest, format_backtest_telegram
            data    = self._report._collect(self._trainer.step_count, self._trainer)
            message = self._report._format_full(data)
            self._report._prev_snapshot = self._report._make_snapshot(self._trainer)
            self._send(message)
            try:
                bt_result = run_backtest(self._trainer.config, trainer=self._trainer)
                self._send(format_backtest_telegram(bt_result))
            except Exception as e:
                self._send(f'⚠️ <i>Backtest error: {e}</i>')
                logger.warning(f'[Commands] Backtest failed: {e}')
        except Exception as e:
            self._send(f'Błąd generowania raportu: {e}')
            logger.warning(f'[Commands] Report error: {e}')

    def _cmd_set(self, args: str):
        """Ustawia wartość zmiennej w configu. Format: sekcja.klucz wartość"""
        parts = args.split(None, 1)
        if len(parts) != 2:
            self._send('Użycie: <code>/set sekcja.klucz wartość</code>\nPrzykład: <code>/set reward.loss_scale 1.5</code>')
            return

        key_path, raw_value = parts[0], parts[1]

        # Rozwiąż ścieżkę w configu (obsługa zagnieżdżonych kluczy: a.b.c)
        keys = key_path.split('.')
        cfg = self._trainer.config
        try:
            node = cfg
            for k in keys[:-1]:
                node = node[k]
            leaf_key = keys[-1]
            if leaf_key not in node:
                raise KeyError(leaf_key)
        except KeyError as e:
            self._send(f'❌ Nieznany klucz: <code>{key_path}</code> (brak sekcji {e})')
            return

        # Wykryj typ na podstawie obecnej wartości
        old_value = node[leaf_key]
        try:
            if isinstance(old_value, bool):
                new_value = raw_value.lower() in ('true', '1', 'yes', 'tak')
            elif isinstance(old_value, int):
                new_value = int(raw_value)
            elif isinstance(old_value, float):
                new_value = float(raw_value)
            else:
                new_value = raw_value
        except ValueError:
            self._send(f'❌ Zła wartość: <code>{raw_value}</code> (oczekiwano {type(old_value).__name__})')
            return

        # Zaktualizuj in-memory config
        node[leaf_key] = new_value

        # Zaktualizuj trainer jeśli to znana zmienna wymagająca propagacji
        self._propagate_to_trainer(key_path, new_value)

        # Zapisz do config.toml
        try:
            self._save_config_toml()
            saved = '✅'
        except Exception as e:
            saved = f'⚠️ Nie zapisano do pliku: {e}'
            logger.warning(f'[Commands] /set save failed: {e}')

        self._send(
            f'{saved} <code>{key_path}</code>\n'
            f'  <b>Było:</b> <code>{old_value}</code>\n'
            f'  <b>Jest:</b>  <code>{new_value}</code>'
        )
        logger.info(f'[Commands] /set {key_path} = {new_value} (było: {old_value})')

    def _cmd_get(self, args: str):
        """Pobiera wartość z configu lub wyświetla całą sekcję. Format: sekcja.klucz lub sekcja"""
        cfg = self._trainer.config

        if not args:
            # Pokaż dostępne sekcje
            sections = [k for k, v in cfg.items() if isinstance(v, dict)]
            self._send('📋 <b>Sekcje konfigu:</b>\n' + '\n'.join(f'  <code>{s}</code>' for s in sections)
                       + '\n\nUżycie: <code>/get sekcja</code> lub <code>/get sekcja.klucz</code>')
            return

        keys = args.split('.')
        try:
            node = cfg
            for k in keys:
                node = node[k]
        except KeyError:
            self._send(f'❌ Nieznany klucz: <code>{args}</code>')
            return

        if isinstance(node, dict):
            # Wyświetl całą sekcję
            lines = [f'📋 <b>[{args}]</b>']
            for k, v in node.items():
                if not isinstance(v, (dict, list)):
                    lines.append(f'  <code>{k}</code> = <code>{v}</code>')
                elif isinstance(v, list) and len(v) <= 5:
                    lines.append(f'  <code>{k}</code> = <code>{v}</code>')
            self._send('\n'.join(lines))
        else:
            self._send(f'📋 <code>{args}</code> = <code>{node}</code>')

    def _propagate_to_trainer(self, key_path: str, value):
        """Propaguje zmiany wybranych kluczy bezpośrednio do trenera."""
        t = self._trainer
        mapping = {
            'training.lr':           lambda v: [pg.update({'lr': v}) for pg in t.optimizer.param_groups],
            'training.epsilon_end':  lambda v: setattr(t, 'epsilon_end', v),
            'training.epsilon_start':lambda v: setattr(t, 'epsilon_start', v),
            'training.gamma':        lambda v: setattr(t, 'gamma', v),
            'training.target_update_interval': lambda v: setattr(t, 'target_update_interval', int(v)),
            'training.checkpoint_interval':    lambda v: setattr(t, 'checkpoint_interval', int(v)),
        }
        if key_path in mapping:
            try:
                mapping[key_path](value)
                logger.info(f'[Commands] Propagated {key_path} to trainer')
            except Exception as e:
                logger.warning(f'[Commands] Propagation failed for {key_path}: {e}')

    def _save_config_toml(self):
        """Zapisuje zmienione wartości do config.toml zachowując komentarze i formatowanie."""
        import tomlkit
        config_path = os.path.abspath(os.path.join(_BASE_DIR, 'config.toml'))
        with open(config_path, 'r', encoding='utf-8') as f:
            doc = tomlkit.load(f)

        # Nakładamy wartości z aktualnego config trainera na dokument tomlkit.
        # Iterujemy tylko po sekcjach skalarnych (nie [[actors]] — to lista tabel).
        skip = {'actors'}
        for section, values in self._trainer.config.items():
            if section in skip or not isinstance(values, dict):
                continue
            if section not in doc:
                doc.add(section, tomlkit.table())
            for key, val in values.items():
                if key in doc[section]:
                    doc[section][key] = val

        with open(config_path, 'w', encoding='utf-8') as f:
            tomlkit.dump(doc, f)

    # ── wysyłanie ─────────────────────────────────────────────────────────────

    def _send(self, text: str) -> None:
        url = f'https://api.telegram.org/bot{self._token}/sendMessage'
        for chunk in _split_message(text):
            try:
                body = urllib.parse.urlencode({
                    'chat_id':    self._chat_id,
                    'text':       chunk,
                    'parse_mode': 'HTML',
                }).encode()
                req = urllib.request.Request(url, data=body, method='POST')
                urllib.request.urlopen(req, timeout=10)
            except Exception as e:
                logger.warning(f'[Commands] Send failed: {e}')
                break
