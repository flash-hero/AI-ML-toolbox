from fastapi.testclient import TestClient
import sys
import os
import pandas as pd
import io

# Add parent dir to sys.path to import api
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Add toolbox to sys.path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'toolbox'))

from api.main import app

client = TestClient(app)

def test_root():
    print("Testing Root...")
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "active"
    print("Root OK")

def test_e2e_flow():
    print("Starting E2E Flow...")
    csv_content = "feature1,feature2,target\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n13,14,15\n16,17,18\n19,20,21\n22,23,24\n25,26,27\n28,29,30"
    filename = "test_data.csv"
    
    # 2. Upload Data
    print("Uploading Data...")
    files = {"file": (filename, io.BytesIO(csv_content.encode('utf-8')), "text/csv")}
    response = client.post("/data/upload", files=files)
    if response.status_code != 200:
        print(f"Upload Failed: {response.text}")
    assert response.status_code == 200
    data_info = response.json()
    assert data_info["filename"] == filename
    assert data_info["rows"] == 10
    print("Upload OK")
    
    # 3. Preview Data
    print("Previewing Data...")
    response = client.get(f"/data/preview/{filename}")
    if response.status_code != 200:
        print(f"Preview Failed: {response.text}")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 5 # Default is 5
    print("Preview OK")

    # ... (rest of the file) ...

if __name__ == "__main__":
    try:
        test_root()
        test_e2e_flow()
        print("All tests passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        import traceback
        traceback.print_exc()

    # 4. Preprocess (Clean)
    print("Preprocessing (Clean)...")
    payload = {
        "filename": filename,
        "impute_strategy": "mean"
    }
    response = client.post("/preprocess/clean", json=payload)
    if response.status_code != 200:
        print(f"Error in /preprocess/clean: Status {response.status_code}")
        print(f"Response Body: {response.text}")
    assert response.status_code == 200
    processed_filename = response.json()["filename"]
    assert processed_filename == f"cleaned_{filename}"
    print("Preprocessing OK")

    # 5. Train Model (Linear Regression)
    print("Training Model...")
    train_payload = {
        "filename": processed_filename,
        "target_column": "target",
        "model_name": "linear_regression",
        "task_type": "regression"
    }
    response = client.post("/train/regression/linear_regression", json=train_payload)
    if response.status_code != 200:
        print(f"Training Failed: {response.text}")
    assert response.status_code == 200
    train_result = response.json()
    assert "model_id" in train_result
    model_id = train_result["model_id"]
    print(f"Trained Model ID: {model_id}")
    print("Training OK")

    # 6. Predict
    print("Predicting...")
    predict_payload = {
        "model_id": model_id,
        "data": [{"feature1": 2, "feature2": 3}]
    }
    response = client.post(f"/predict/{model_id}", json=predict_payload, params={"model_id": model_id})
    if response.status_code != 200:
        print(f"Prediction Failed: {response.text}")
    
    assert response.status_code == 200
    predictions = response.json()["predictions"]
    assert len(predictions) == 1
    print(f"Prediction: {predictions[0]}")
    print("Prediction OK")
    
    # Clean up (optional)
    # os.remove(os.path.join("uploaded_data", filename))
    # os.remove(os.path.join("processed_data", processed_filename))

if __name__ == "__main__":
    try:
        test_root()
        test_e2e_flow()
        print("All tests passed!")
    except Exception as e:
        print(f"Tests failed: {e}")
        # import traceback
        # traceback.print_exc()
