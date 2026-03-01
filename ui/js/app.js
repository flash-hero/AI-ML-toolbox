const app = {
    state: {
        currentFile: null,
        metadata: null,
        trainedModel: null,
        targetColumn: null,
        taskType: null
    },

    init() {
        const navItems = document.querySelectorAll('.nav-item');
        navItems.forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                navItems.forEach(n => n.classList.remove('active'));
                item.classList.add('active');
                this.navigateTo(item.getAttribute('data-page'));
            });
        });
        this.navigateTo('home');
    },

    navigateTo(page) {
        switch (page) {
            case 'home': ui.renderHome(); break;
            case 'upload': ui.renderUpload(); break;
            case 'preprocessing': ui.renderPreprocessing(); break;
            case 'training': ui.renderTraining(); break;
            case 'prediction': ui.renderPrediction(); break;
        }
    },

    async handleFileUpload(file) {
        if (!file) return;
        ui.showLoader(true);
        try {
            const result = await api.uploadData(file);
            this.state.currentFile = result.filename;
            this.state.metadata = result;
            const previewData = await api.previewData(result.filename);
            this.updatePreviewTable(previewData);
            const infoDiv = document.getElementById('dataset-info');
            if (infoDiv) infoDiv.innerText = `Uploaded: ${result.filename} (${result.rows} rows, ${result.columns} columns)`;
        } catch (err) {
            alert("Error uploading file: " + err.message);
        } finally {
            ui.showLoader(false);
        }
    },

    updatePreviewTable(dataObj) {
        const table = document.getElementById('preview-table');
        const previewArea = document.getElementById('preview-area');
        let html = '<thead><tr>' + dataObj.columns.map(c => `<th>${c}</th>`).join('') + '</tr></thead>';
        html += '<tbody>';
        dataObj.data.forEach(row => {
            html += '<tr>' + dataObj.columns.map(c => `<td>${row[c]}</td>`).join('') + '</tr>';
        });
        html += '</tbody>';
        table.innerHTML = html;
        previewArea.style.display = 'block';
    },

    async cleanData() {
        const strategy = document.getElementById('impute-strategy').value;
        const unchecked = document.querySelectorAll('input[name="selected-cols"]:not(:checked)');
        const dropCols = Array.from(unchecked).map(cb => cb.value);

        ui.showLoader(true);
        try {
            const result = await api.cleanData(this.state.currentFile, strategy, dropCols);
            this.state.currentFile = result.filename;
            this.state.metadata = result;
            alert(`Data Cleaned! Dropped ${dropCols.length} columns.`);
            ui.renderPreprocessing();
        } catch (err) {
            alert("Cleaning failed: " + err.message);
        } finally {
            ui.showLoader(false);
        }
    },

    async processDataUnified() {
        const autoSelect = document.getElementById('auto-select-features').checked;
        const isTimeSeries = document.getElementById('is-time-series').checked;
        const selectionTarget = document.getElementById('selection-target-col').value;
        const dateCol = document.getElementById('date-col').value;

        // Basic inputs
        const strategy = document.getElementById('impute-strategy').value;
        const scaleMethod = document.getElementById('scale-method').value;
        const encodeMethod = document.getElementById('encode-method').value;

        // If feature selection or target encoding is on, we need a target
        const needsTarget = autoSelect || encodeMethod === 'target';
        const targetCol = needsTarget ? selectionTarget : null;

        ui.showLoader(true);
        try {
            // Call the new unified endpoint in API (need to add client method too)
            const result = await api.processUnified({
                filename: this.state.currentFile,
                impute_strategy: strategy,
                scale_method: scaleMethod,
                encode_method: encodeMethod,

                auto_select_features: autoSelect,
                target_column: targetCol, // Used for both selection and target encoding

                is_time_series: isTimeSeries,
                date_column: dateCol || null
            });

            this.state.currentFile = result.filename;
            this.state.metadata = result;
            this.state.isTimeSeries = isTimeSeries; // Track state

            alert(`Data Processed Successfully!\nRows: ${result.rows}, Cols: ${result.columns}`);
            ui.renderPreprocessing();
        } catch (err) {
            alert("Processing failed: " + err.message);
        } finally {
            ui.showLoader(false);
        }
    },

    async scaleData() {
        const method = document.getElementById('scale-method').value;
        ui.showLoader(true);
        try {
            const result = await api.scaleData(this.state.currentFile, method);
            this.state.currentFile = result.filename;
            alert(`Features Scaled using ${method}!`);
        } catch (err) {
            alert("Scaling failed: " + err.message);
        } finally {
            ui.showLoader(false);
        }
    },

    async encodeData() {
        const method = document.getElementById('encode-method').value;
        let targetCol = null;
        if (method === 'target') {
            targetCol = prompt("Enter the Target Column name for Target Encoding:");
            if (!targetCol) return;
        }
        ui.showLoader(true);
        try {
            const result = await api.encodeData(this.state.currentFile, method, targetCol);
            this.state.currentFile = result.filename;
            this.state.metadata = result;
            alert(`Categorical Variables Encoded using ${method}!`);
            ui.renderPreprocessing();
        } catch (err) {
            alert("Encoding failed: " + err.message);
        } finally {
            ui.showLoader(false);
        }
    },

    async generateVisualizations() {
        ui.showLoader(true);
        try {
            const result = await api.generateVisualizations(this.state.currentFile);
            const container = document.getElementById('viz-container');
            container.innerHTML = ''; // Clear old

            // Loop through plots
            for (const [name, url] of Object.entries(result.plots)) {
                const fullUrl = 'http://127.0.0.1:8000' + url;
                const div = document.createElement('div');
                div.innerHTML = `
                    <div style="background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                        <h4 style="text-align:center; color: var(--action-color); text-transform: capitalize;">${name}</h4>
                        <img src="${fullUrl}" style="width: 100%; border-radius: 4px;" alt="${name}">
                    </div>
                `;
                container.appendChild(div);
            }
        } catch (err) {
            alert("Visualization Error: " + err.message);
        } finally {
            ui.showLoader(false);
        }
    },

    async trainModel() {
        const taskType = document.getElementById('task-type').value;
        const modelName = document.getElementById('model-name').value;
        const targetCol = document.getElementById('target-column') ? document.getElementById('target-column').value : null;

        ui.showLoader(true);
        try {
            const result = await api.trainModel(
                this.state.currentFile,
                targetCol,
                taskType,
                modelName,
                this.state.isTimeSeries || false // Pass time series flag
            );

            const resultsDiv = document.getElementById('training-results');
            resultsDiv.style.display = 'block';

            let resultHtml = `<h3>Training Complete</h3>`;
            if (result.model_id) {
                this.state.trainedModel = result.model_id;
                this.state.targetColumn = targetCol;
                this.state.taskType = taskType;

                resultHtml += `<p style="color: green;"> <i class="ph ph-check-circle"></i> Model Saved: ${result.model_id}</p>`;

                // Display Metrics
                if (result.test_score) resultHtml += `<p><b>R2 Score:</b> ${result.test_score}</p>`;
                if (result.test_accuracy) resultHtml += `<p><b>Accuracy:</b> ${result.test_accuracy}</p>`;

                // --- NEW: Display Plots ---
                if (result.plots && result.plots.length > 0) {
                    resultHtml += `<div style="display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); margin-top: 20px;">`;
                    result.plots.forEach(plotUrl => {
                        const fullUrl = 'http://127.0.0.1:8000' + plotUrl;
                        resultHtml += `
                            <div style="background: white; padding: 10px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                                <img src="${fullUrl}" style="width: 100%;" alt="Evaluation Plot">
                            </div>
                         `;
                    });
                    resultHtml += `</div>`;
                }

                resultHtml += `<br><button class="btn" onclick="app.navigateTo('prediction')">Go to Prediction</button>`;
            } else {
                resultHtml += `<p>${result.message}</p>`;
            }
            resultsDiv.innerHTML = resultHtml;

        } catch (err) {
            alert("Training failed: " + err.message);
        } finally {
            ui.showLoader(false);
        }
    },

    async makePrediction() {
        const inputs = {};
        const inputElements = document.querySelectorAll('.feature-input');
        inputElements.forEach(input => {
            const col = input.getAttribute('data-col');
            const val = Number(input.value);
            inputs[col] = isNaN(val) ? input.value : val;
        });

        ui.showLoader(true);
        try {
            const result = await api.predict(this.state.trainedModel, [inputs]);
            const resultDiv = document.getElementById('prediction-result');
            resultDiv.style.display = 'block';
            document.getElementById('pred-value').innerText = result.predictions[0];
        } catch (err) {
            alert("Prediction Error: " + err.message);
        } finally {
            ui.showLoader(false);
        }
    }
};


document.addEventListener('DOMContentLoaded', () => {
    app.init();

    // Event delegation for dynamic content
    document.body.addEventListener('change', (e) => {
        if (e.target.id === 'auto-select-features') {
            document.getElementById('target-col-div').style.display = e.target.checked ? 'block' : 'none';
        }
        if (e.target.id === 'is-time-series') {
            document.getElementById('date-col-div').style.display = e.target.checked ? 'block' : 'none';
        }
    });
});
