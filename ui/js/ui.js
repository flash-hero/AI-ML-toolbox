const ui = {
    // Selectors
    contentContainer: document.getElementById('page-content'),

    renderHome() {
        this.contentContainer.innerHTML = `
            <div class="page-header">
                <h1>Welcome Back</h1>
                <p>Advanced AI Toolbox for Regression, Classification, and Clustering.</p>
            </div>
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">3</div>
                    <div class="stat-label">Model Types</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">12+</div>
                    <div class="stat-label">Algorithms</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">Fast</div>
                    <div class="stat-label">Performance</div>
                </div>
            </div>
            <div class="card">
                <h3>Getting Started</h3>
                <p>Start by uploading your dataset in the <b>Upload Data</b> section. Then clean it efficiently before training your models.</p>
                <br>
                <button class="btn" onclick="app.navigateTo('upload')">Start New Project</button>
            </div>
        `;
    },

    renderUpload() {
        this.contentContainer.innerHTML = `
            <div class="page-header">
                <h1>Upload Data</h1>
                <p>Import your dataset (CSV or Excel) to begin.</p>
            </div>
            
            <div class="card">
                <div class="upload-zone" id="drop-zone">
                    <i class="ph ph-file-arrow-up upload-icon"></i>
                    <h3>Drag & Drop your file here</h3>
                    <p>or click to browse</p>
                    <input type="file" id="file-input" hidden accept=".csv, .xlsx">
                </div>
            </div>

            <div id="preview-area" class="card" style="display: none;">
                <h3>Data Preview</h3>
                <div id="dataset-info" style="margin-bottom: 1em; color: var(--text-secondary);"></div>
                <div class="data-table-container">
                    <table class="data-table" id="preview-table">
                        <!-- Rows injected here -->
                    </table>
                </div>
                <br>
                <button class="btn" onclick="app.navigateTo('preprocessing')">Proceed to Preprocessing <i class="ph ph-arrow-right"></i></button>
            </div>
        `;

        // Add event listeners for this specific page
        const dropZone = document.getElementById('drop-zone');
        const fileInput = document.getElementById('file-input');

        dropZone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', (e) => app.handleFileUpload(e.target.files[0]));

        // Drag over effects
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.style.backgroundColor = 'rgba(86, 28, 36, 0.1)';
        });
        dropZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dropZone.style.backgroundColor = 'rgba(255, 255, 255, 0.3)';
        });
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.style.backgroundColor = 'rgba(255, 255, 255, 0.3)';
            if (e.dataTransfer.files.length) {
                app.handleFileUpload(e.dataTransfer.files[0]);
            }
        });
    },

    renderPreprocessing() {
        if (!app.state.currentFile) {
            this.contentContainer.innerHTML = `
                <div class="page-header"><h1>Preprocessing</h1></div>
                <div class="card">
                    <p>No data loaded. Please upload a file first.</p>
                    <button class="btn" onclick="app.navigateTo('upload')">Go to Upload</button>
                </div>`;
            return;
        }

        // Generate Column Selection Checkboxes
        const columns = app.state.metadata ? app.state.metadata.column_names : [];
        const columnCheckboxes = columns.map(col => `
            <div style="display: inline-block; margin-right: 15px; margin-bottom: 5px;">
                <input type="checkbox" id="col-${col}" name="selected-cols" value="${col}" checked>
                <label for="col-${col}">${col}</label>
            </div>
        `).join('');


        this.contentContainer.innerHTML = `
            <div class="page-header">
                <h1>Preprocessing</h1>
                <p>Working on: <b>${app.state.currentFile}</b></p>
            </div>

            <div class="card">
                <h3>1. Select Columns to Keep</h3>
                <p style="font-size: 0.9em; color: var(--text-secondary); margin-bottom: 10px;">Uncheck columns you want to drop.</p>
                <div style="max-height: 150px; overflow-y: auto; padding: 10px; border: 1px solid #ccc; border-radius: 6px; margin-bottom: 20px;">
                    ${columnCheckboxes}
                </div>
            </div>

            <div class="card">
                <h3>2. Clean Data</h3>
                <div class="form-group">
                    <label class="form-label">Missing Values Strategy</label>
                    <select class="form-select" id="impute-strategy">
                        <option value="mean">Mean (Average)</option>
                        <option value="median">Median (Middle Value)</option>
                        <option value="mode">Mode (Most Frequent)</option>
                        <option value="drop">Drop Rows</option>
                    </select>
                </div>
                <button class="btn" onclick="app.cleanData()">Apply Cleaning</button>
            </div>

            <div class="card">
                <h3>3. Advanced Options</h3>
                
                <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 20px;">
                    <div style="flex: 1; min-width: 250px;">
                        <input type="checkbox" id="auto-select-features">
                        <label for="auto-select-features"><b>Auto-Select Best Features</b></label>
                        <p style="font-size: 0.8em; color: var(--text-secondary);">Automatically picks the most important columns.</p>
                        <div id="target-col-div" style="display:none; margin-top:10px;">
                           <label class="form-label">Target Column (for selection)</label>
                           <select class="form-select" id="selection-target-col">
                               ${app.state.metadata.column_names.map(c => `<option value="${c}">${c}</option>`).join('')}
                           </select>
                        </div>
                    </div>

                    <div style="flex: 1; min-width: 250px;">
                        <input type="checkbox" id="is-time-series">
                        <label for="is-time-series"><b>Time Series Data</b></label>
                        <p style="font-size: 0.8em; color: var(--text-secondary);">Preserves order for RNN/LSTM models.</p>
                        <div id="date-col-div" style="display:none; margin-top:10px;">
                           <label class="form-label">Date Column</label>
                           <select class="form-select" id="date-col">
                               <option value="">Auto-Detect</option>
                               ${app.state.metadata.column_names.map(c => `<option value="${c}">${c}</option>`).join('')}
                           </select>
                        </div>
                    </div>
                </div>

                <button class="btn" onclick="app.processDataUnified()">Run Advanced Processing</button>
            </div>

            <div class="card">
                 <h3>4. Manual Scaling & Encoding</h3>
                 <div style="display: flex; gap: 20px; flex-wrap: wrap;">
                     <div style="flex: 1; min-width: 250px;">
                        <label class="form-label">Scaling Method</label>
                        <select class="form-select" id="scale-method" style="margin-bottom: 10px;">
                            <option value="standard">Standard Scaler (Mean 0, Std 1)</option>
                            <option value="minmax">MinMax Scaler (0 to 1)</option>
                        </select>
                        <button class="btn btn-secondary" onclick="app.scaleData()">Apply Scaling Only</button>
                     </div>

                     <div style="flex: 1; min-width: 250px;">
                        <label class="form-label">Encoding Method</label>
                        <select class="form-select" id="encode-method" style="margin-bottom: 10px;">
                            <option value="onehot">One-Hot Encoding</option>
                            <option value="label">Label Encoding</option>
                            <option value="target">Target Encoding</option>
                        </select>
                        <button class="btn btn-secondary" onclick="app.encodeData()">Apply Encoding Only</button>
                     </div>
                 </div>
            </div>

            <div class="card">
                <h3>5. Visualization</h3>
                <p>Generate plots to understand your data distribution and correlations.</p>
                <button class="btn" onclick="app.generateVisualizations()" id="btn-vis">Generate Plots</button>
                <div id="viz-container" style="margin-top: 20px; display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));">
                    <!-- Images injected here -->
                </div>
            </div>
            
            <div id="preprocessing-status" class="card" style="display: none;"></div>
        `;
    },

    renderTraining() {
        if (!app.state.currentFile) {
            this.contentContainer.innerHTML = `
                <div class="page-header"><h1>Training</h1></div>
                <div class="card"><p>Please upload and preprocess data first.</p></div>`;
            return;
        }

        const columns = app.state.metadata ? app.state.metadata.column_names : [];
        const options = columns.map(c => `<option value="${c}">${c}</option>`).join('');

        this.contentContainer.innerHTML = `
            <div class="page-header">
                <h1>Model Training</h1>
                <p>Train your AI model on <b>${app.state.currentFile}</b></p>
            </div>

            <div class="card">
                <div class="form-group">
                    <label class="form-label">Task Type</label>
                    <select class="form-select" id="task-type" onchange="ui.updateModelOptions()">
                        <option value="regression">Regression (Predict Numbers)</option>
                        <option value="classification">Classification (Predict Categories)</option>
                        <option value="clustering">Clustering (Group Data)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label class="form-label">Algorithm</label>
                    <select class="form-select" id="model-name">
                        <option value="linear_regression">Linear Regression</option>
                        <!-- Updated by JS based on Task Type -->
                    </select>
                </div>

                <div class="form-group" id="target-col-group">
                    <label class="form-label">Target Column (What to predict)</label>
                    <select class="form-select" id="target-column">
                        ${options}
                    </select>
                </div>

                <button class="btn" onclick="app.trainModel()">Start Training</button>
            </div>

            <div id="training-results" class="card" style="display: none;">
                <!-- Results injected here -->
            </div>
        `;
    },

    updateModelOptions() {
        const taskType = document.getElementById('task-type').value;
        const modelSelect = document.getElementById('model-name');
        const targetGroup = document.getElementById('target-col-group');

        let models = [];
        if (taskType === 'regression') {
            models = [
                { val: 'linear', name: 'Linear Regression' },
                { val: 'random_forest', name: 'Random Forest Regressor' },
                { val: 'lstm', name: 'LSTM (Time Series Only)' },
                // Add more...
            ];
            targetGroup.style.display = 'block';
        } else if (taskType === 'classification') {
            models = [
                { val: 'logistic', name: 'Logistic Regression' },
                { val: 'random_forest', name: 'Random Forest' },
                { val: 'knn', name: 'K-Nearest Neighbors (KNN)' },
                { val: 'decision_tree', name: 'Decision Tree' },
                { val: 'svm', name: 'Support Vector Machine (SVM)' },
                { val: 'naive_bayes', name: 'Naive Bayes' },
                { val: 'xgboost', name: 'XGBoost' },
                { val: 'lstm', name: 'LSTM (Time Series Only)' },
                { val: 'gru', name: 'GRU (Time Series Only)' }
            ];
            targetGroup.style.display = 'block';
        } else if (taskType === 'clustering') {
            models = [{ val: 'kmeans', name: 'K-Means Clustering' }];
            targetGroup.style.display = 'none';
        }

        modelSelect.innerHTML = models.map(m => `<option value="${m.val}">${m.name}</option>`).join('');
    },

    renderPrediction() {
        if (!app.state.trainedModel) {
            this.contentContainer.innerHTML = `
                <div class="page-header"><h1>Prediction</h1></div>
                <div class="card"><p>No model trained yet. Please train a model first.</p></div>`;
            return;
        }

        // Generate inputs for features
        const inputFields = app.state.metadata.column_names
            .filter(c => c !== app.state.targetColumn) // Exclude target
            .map(c => `
                <div class="form-group">
                    <label class="form-label">${c}</label>
                    <input type="text" class="form-input feature-input" data-col="${c}" placeholder="Enter value">
                </div>
            `).join('');

        this.contentContainer.innerHTML = `
            <div class="page-header">
                <h1>Make Prediction</h1>
                <p>Using model: <b>${app.state.trainedModel}</b></p>
            </div>

            <div class="card">
                <h3>Enter Data Points</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px;">
                    ${inputFields}
                </div>
                <br>
                <button class="btn" onclick="app.makePrediction()">Predict</button>
            </div>

            <div id="prediction-result" class="card" style="display: none; text-align: center;">
                 <h3>Result</h3>
                 <div class="stat-value" id="pred-value">-</div>
            </div>
        `;
    },

    showLoader(show = true) {
        document.body.style.cursor = show ? 'wait' : 'default';
        const btns = document.querySelectorAll('button');
        btns.forEach(b => b.disabled = show);
    }
};
