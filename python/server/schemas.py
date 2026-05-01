"""
Schematy wiadomości MessagePack dla komunikacji ZMQ.
Definiuje format danych wymienianych między Actor (Node.js) a Learner (Python).
"""


def validate_predict_request(data):
    if 'state' not in data:
        raise ValueError("Missing field in predict request: state")

    state = data['state']
    if not isinstance(state, dict):
        raise ValueError("State must be a dict with timeframe keys")

    return True


def validate_batch_request(data):
    if not isinstance(data, list):
        raise ValueError("Batch request must be a list")

    for i, item in enumerate(data):
        if 'actorId' not in item:
            raise ValueError(f"Missing actorId in batch item {i}")
        if 'state' not in item:
            raise ValueError(f"Missing state in batch item {i}")

    return True


def build_step_response(action):
    return {'nextAction': int(action)}


def build_predict_response(action, q_values, epsilon=None):
    resp = {
        'action': int(action),
        'qValues': q_values.tolist() if hasattr(q_values, 'tolist') else list(q_values)
    }
    if epsilon is not None:
        resp['epsilon'] = float(epsilon)
    return resp


