import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'

def test_predict_baseline():
    r = client.post('/predict', json={'period_dates': ['2025-06-01','2025-06-30','2025-07-28']})
    assert r.status_code == 200
    data = r.json()
    assert 'next_period_date' in data
    assert data['predicted_cycle_length'] > 0
