import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import './index.css';

const errEl = document.createElement('div');
errEl.id = 'fatal-error';
document.body.appendChild(errEl);
const showFatal = (msg) => { errEl.style.display = 'block'; errEl.textContent = msg; };
window.addEventListener('error', (e) => showFatal(`JS ERROR\n${e.message}\n${e.filename}:${e.lineno}\n\n${e.error?.stack ?? ''}`));
window.addEventListener('unhandledrejection', (e) => showFatal(`Unhandled Promise: ${e.reason}`));

createRoot(document.getElementById('root')).render(
  <StrictMode><App /></StrictMode>
);
