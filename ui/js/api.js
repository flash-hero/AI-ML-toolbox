const API_BASE_URL = 'http://127.0.0.1:8000';

const api = {
    // --- Data Management ---
    async uploadData(file) {
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch(`${API_BASE_URL}/data/upload`, {
                method: 'POST',
                body: formData
            });
            if (!response.ok) throw new Error(await response.text());
            return await response.json();
        } catch (error) {
            console.error("Upload failed", error);
            throw error;
        }
    },

    async previewData(filename) {
        try {
            const response = await fetch(`${API_BASE_URL}/data/preview/${filename}`);
            if (!response.ok) throw new Error(await response.text());
            return await response.json();
        } catch (error) {
            console.error("Preview failed", error);
            throw error;
        }
    },

    // --- Preprocessing ---
    async cleanData(filename, strategy = 'mean', dropColumns = []) {
        try {
            const response = await fetch(`${API_BASE_URL}/preprocess/clean`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename,
                    impute_strategy: strategy,
                    drop_columns: dropColumns
                })
            });
            if (!response.ok) throw new Error(await response.text());
            return await response.json();
        } catch (error) {
            console.error("Cleaning failed", error);
            throw error;
        }
    },

    async scaleData(filename, method = 'standard') {
        try {
            const response = await fetch(`${API_BASE_URL}/preprocess/scale`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ filename, scale_method: method })
            });
            if (!response.ok) throw new Error(await response.text());
            return await response.json();
        } catch (error) {
            console.error("Scaling failed", error);
            throw error;
        }
    },

    async encodeData(filename, method = 'onehot', targetCol = null) {
        try {
            const response = await fetch(`${API_BASE_URL}/preprocess/encode`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename,
                    encode_method: method,
                    target_column: targetCol
                })
            });
            if (!response.ok) throw new Error(await response.text());
            return await response.json();
        } catch (error) {
            console.error("Encoding failed", error);
            throw error;
        }
    },

    async processUnified(params) {
        try {
            const response = await fetch(`${API_BASE_URL}/preprocess/process`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            if (!response.ok) throw new Error(await response.text());
            return await response.json();
        } catch (error) {
            console.error("Processing failed", error);
            throw error;
        }
    },

    // --- Visualization ---
    async generateVisualizations(filename) {
        try {
            const response = await fetch(`${API_BASE_URL}/visualization/generate/${filename}`);
            if (!response.ok) throw new Error(await response.text());
            return await response.json();
        } catch (error) {
            console.error("Visualization failed", error);
            throw error;
        }
    },


    // --- Training ---
    async trainModel(filename, targetCol, taskType, modelName, isTimeSeries = false) {
        try {
            const response = await fetch(`${API_BASE_URL}/train/${taskType}/${modelName}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filename,
                    target_column: targetCol,
                    model_name: modelName,
                    task_type: taskType,
                    is_time_series: isTimeSeries
                })
            });
            if (!response.ok) throw new Error(await response.text());
            return await response.json();
        } catch (error) {
            console.error("Training failed", error);
            throw error;
        }
    },

    // --- Prediction ---
    async predict(modelId, data) {
        try {
            const response = await fetch(`${API_BASE_URL}/predict/${modelId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model_id: modelId,
                    data: data
                })
            });
            if (!response.ok) throw new Error(await response.text());
            return await response.json();
        } catch (error) {
            console.error("Prediction failed", error);
            throw error;
        }
    }
};
