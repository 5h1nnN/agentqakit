def run(payload):
    return {'answer': payload.get('input', {}).get('message', '')}
