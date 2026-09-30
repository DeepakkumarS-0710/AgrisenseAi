document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const imageInput = document.getElementById('image-input');
    const imagePreview = document.getElementById('image-preview');
    const uploadPlaceholder = document.getElementById('upload-placeholder');
    const analyzeForm = document.getElementById('analyze-form');
    const analyzeBtn = document.getElementById('analyze-btn');
    const analyzeSpinner = document.getElementById('analyze-spinner');
    const btnText = document.getElementById('btn-text');
    const syncIotBtn = document.getElementById('sync-iot-btn');

    // 1. Image Drag & Drop Preview Setup
    if (dropZone && imageInput) {
        dropZone.addEventListener('click', () => imageInput.click());

        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });

        ['dragleave', 'dragend'].forEach(type => {
            dropZone.addEventListener(type, () => dropZone.classList.remove('dragover'));
        });

        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                imageInput.files = e.dataTransfer.files;
                handleImagePreview(e.dataTransfer.files[0]);
            }
        });

        imageInput.addEventListener('change', () => {
            if (imageInput.files && imageInput.files.length > 0) {
                handleImagePreview(imageInput.files[0]);
            }
        });
    }

    function handleImagePreview(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file (JPG, PNG, WEBP).');
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            uploadPlaceholder.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    // 2. Sync IoT Latest Sensor Telemetry
    if (syncIotBtn) {
        syncIotBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/api/sensors/latest');
                const result = await response.json();
                if (result.status === 'success' && result.data) {
                    const data = result.data;
                    document.getElementById('temperature').value = data.temperature || 25.5;
                    document.getElementById('humidity').value = data.humidity || 72.0;
                    document.getElementById('ph').value = data.ph || 6.5;
                    document.getElementById('nitrogen').value = data.nitrogen || 50;
                    document.getElementById('phosphorus').value = data.phosphorus || 50;
                    document.getElementById('potassium').value = data.potassium || 50;
                    document.getElementById('rainfall').value = data.rainfall || 100.0;

                    if (data.soil_moisture && data.soil_moisture !== 'Not Available') {
                        document.getElementById('soil_moisture').value = data.soil_moisture;
                        document.getElementById('card-val-sm').innerText = data.soil_moisture;
                        document.getElementById('card-badge-sm').innerText = 'Normal';
                        document.getElementById('card-badge-sm').className = 'status-badge normal';
                    }

                    // Update sensor telemetry cards
                    updateSensorCard('temp', data.temperature);
                    updateSensorCard('hum', data.humidity);
                    updateSensorCard('ph', data.ph);
                    updateSensorCard('n', data.nitrogen);
                    updateSensorCard('p', data.phosphorus);
                    updateSensorCard('k', data.potassium);
                    updateSensorCard('rain', data.rainfall);

                    alert('Synced latest IoT sensor telemetry successfully!');
                } else {
                    alert('No IoT sensor data found yet. Enter manual values in form.');
                }
            } catch (err) {
                console.error('Failed to sync IoT telemetry:', err);
            }
        });
    }

    function updateSensorCard(key, value) {
        const el = document.getElementById(`card-val-${key}`);
        if (el && value !== undefined) {
            el.innerText = value;
        }
    }

    // 3. Form Submit & AJAX Fetch Analysis
    if (analyzeForm) {
        analyzeForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            // Client-side Validation
            const temp = parseFloat(document.getElementById('temperature').value);
            const hum = parseFloat(document.getElementById('humidity').value);
            const ph = parseFloat(document.getElementById('ph').value);

            if (isNaN(temp) || isNaN(hum) || isNaN(ph)) {
                alert('Please enter valid numeric values for Temperature, Humidity, and pH.');
                return;
            }

            // Show Loading State
            analyzeBtn.disabled = true;
            analyzeSpinner.style.display = 'inline-block';
            btnText.innerText = 'RUNNING MULTIMODAL ANALYSIS...';

            const formData = new FormData(analyzeForm);

            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });

                const res = await response.json();

                if (res.status === 'success' && res.data) {
                    renderAnalysisResults(res.data);
                } else {
                    alert(`Analysis Error: ${res.message || 'Unknown error occurred.'}`);
                }

            } catch (err) {
                alert(`Network error during analysis: ${err.message}`);
            } finally {
                analyzeBtn.disabled = false;
                analyzeSpinner.style.display = 'none';
                btnText.innerText = 'ANALYZE MULTIMODAL DATA';
            }
        });
    }

    // 4. Render Dynamic Multimodal Results to DOM
    function renderAnalysisResults(data) {
        const crop = data.crop_recommendation || {};
        const disease = data.disease_detection || {};
        const advisor = data.advisor || {};
        const ragSources = data.rag_sources || [];

        // Crop Recommendation
        document.getElementById('res-crop').innerText = (crop.recommended_crop || '--').toUpperCase();
        document.getElementById('res-crop-conf').innerText = `${crop.confidence || 0}%`;
        const inputs = crop.input_summary || {};
        document.getElementById('res-npk-summary').innerText = `N:${inputs.N || '--'} P:${inputs.P || '--'} K:${inputs.K || '--'}`;
        document.getElementById('res-env-summary').innerText = `Temp ${inputs.temperature || '--'}°C, Humidity ${inputs.humidity || '--'}%, pH ${inputs.ph || '--'}`;

        // Plant Disease Detection
        const healthStatus = disease.health_status || 'Unknown';
        const healthEl = document.getElementById('res-health-status');
        healthEl.innerText = healthStatus.toUpperCase();
        if (healthStatus.toLowerCase() === 'healthy') {
            healthEl.style.color = '#34d399';
        } else if (healthStatus.toLowerCase() === 'diseased') {
            healthEl.style.color = '#f87171';
        } else {
            healthEl.style.color = '#94a3b8';
        }

        document.getElementById('res-disease').innerText = disease.disease || '--';
        document.getElementById('res-disease-conf').innerText = `${disease.confidence || 0}%`;
        document.getElementById('res-disease-crop').innerText = disease.crop || '--';

        // Risk Level Badge
        const riskLevel = advisor.risk_level || 'Low';
        const riskBadge = document.getElementById('res-risk-level');
        riskBadge.innerText = `Risk: ${riskLevel}`;
        if (riskLevel.toLowerCase() === 'high' || riskLevel.toLowerCase() === 'critical') {
            riskBadge.className = 'status-badge critical';
        } else if (riskLevel.toLowerCase() === 'medium') {
            riskBadge.className = 'status-badge high';
        } else {
            riskBadge.className = 'status-badge normal';
        }

        // Advisor Recommendations
        document.getElementById('res-overall-assessment').innerText = advisor.overall_assessment || 'No assessment generated.';
        document.getElementById('res-disease-exp').innerText = advisor.disease_explanation || '--';
        document.getElementById('res-treatment').innerText = advisor.treatment_recommendation || '--';
        document.getElementById('res-fertilizer').innerText = advisor.fertilizer_recommendation || '--';
        document.getElementById('res-irrigation').innerText = advisor.irrigation_recommendation || '--';
        document.getElementById('res-soil').innerText = advisor.soil_management || '--';
        document.getElementById('res-prevention').innerText = advisor.prevention || '--';

        // RAG Sources
        const ragContainer = document.getElementById('rag-sources-list');
        if (ragSources && ragSources.length > 0) {
            ragContainer.innerHTML = ragSources.map(item => `
                <div class="rag-card">
                    <div class="rag-doc-title">📄 Document: ${item.source_doc} (Relevance Score: ${item.score})</div>
                    <div class="rag-snippet">"${item.text}"</div>
                </div>
            `).join('');
        } else {
            ragContainer.innerHTML = '<div style="color: var(--text-muted); font-size: 0.9rem;">No specific RAG sources retrieved.</div>';
        }

        // Smooth scroll down to results section
        document.querySelector('.results-container').scrollIntoView({ behavior: 'smooth' });
    }
});
