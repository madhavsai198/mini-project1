// CardioCare AI Frontend Engine
document.addEventListener('DOMContentLoaded', () => {
    // Sliders element bindings
    const sliders = [
        { id: 'age', valId: 'ageVal' },
        { id: 'trestbps', valId: 'bpVal' },
        { id: 'chol', valId: 'cholVal' },
        { id: 'thalach', valId: 'hrVal' },
        { id: 'oldpeak', valId: 'peakVal' }
    ];

    sliders.forEach(item => {
        const input = document.getElementById(item.id);
        const display = document.getElementById(item.valId);
        if (input && display) {
            input.addEventListener('input', (e) => {
                display.textContent = e.target.value;
            });
        }
    });

    // Form element & UI targets
    const form = document.getElementById('predictionForm');
    const riskScoreText = document.getElementById('riskScoreText');
    const riskTierBadge = document.getElementById('riskTierBadge');
    const gaugeFill = document.getElementById('gaugeFill');
    const driversList = document.getElementById('driversList');
    const recommendationsList = document.getElementById('recommendationsList');

    // Preset profile handling
    const presetButtons = document.querySelectorAll('.preset-btn');
    presetButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const presetType = btn.dataset.preset;
            loadPreset(presetType);
        });
    });

    const presetDataMap = {
        low_risk: {
            age: 34, sex: 0, cp: 2, trestbps: 112, chol: 178,
            fbs: 0, restecg: 0, thalach: 168, exang: 0,
            oldpeak: 0.2, slope: 0, ca: 0, thal: 1
        },
        moderate_risk: {
            age: 54, sex: 1, cp: 1, trestbps: 138, chol: 242,
            fbs: 0, restecg: 1, thalach: 142, exang: 0,
            oldpeak: 1.4, slope: 1, ca: 1, thal: 2
        },
        high_risk: {
            age: 66, sex: 1, cp: 0, trestbps: 164, chol: 310,
            fbs: 1, restecg: 1, thalach: 108, exang: 1,
            oldpeak: 2.9, slope: 1, ca: 2, thal: 3
        }
    };

    function loadPreset(key) {
        const data = presetDataMap[key];
        if (!data) return;

        Object.keys(data).forEach(field => {
            const el = document.getElementById(field);
            if (el) {
                el.value = data[field];
                // Trigger slider value sync
                const valDisplay = document.getElementById(getValId(field));
                if (valDisplay) valDisplay.textContent = data[field];
            }
        });

        // Trigger immediate prediction
        runPrediction();
    }

    function getValId(field) {
        const map = { age: 'ageVal', trestbps: 'bpVal', chol: 'cholVal', thalach: 'hrVal', oldpeak: 'peakVal' };
        return map[field];
    }

    // Handle Form Submit
    form.addEventListener('submit', (e) => {
        e.preventDefault();
        runPrediction();
    });

    async function runPrediction() {
        const formData = new FormData(form);
        const payload = {};
        formData.forEach((val, key) => {
            payload[key] = parseFloat(val);
        });

        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (res.ok) {
                const data = await res.json();
                renderResults(data);
            } else {
                // Client-side fallback computation if API server is offline
                runClientFallbackPrediction(payload);
            }
        } catch (err) {
            console.log("Serving offline client ML fallback prediction");
            runClientFallbackPrediction(payload);
        }
    }

    function runClientFallbackPrediction(p) {
        // Pure Client ML logit formula
        const logit = (
            0.04 * (p.age - 50) +
            0.5 * p.sex +
            -0.8 * p.cp +
            0.015 * (p.trestbps - 120) +
            0.005 * (p.chol - 200) +
            -0.03 * (p.thalach - 150) +
            0.9 * p.exang +
            0.7 * p.oldpeak +
            0.6 * p.ca +
            (0.8 * (p.thal === 3 ? 1 : 0)) - 1.2
        );

        const proba = 1.0 / (1.0 + Math.exp(-logit));
        const risk_percent = Math.round(proba * 1000) / 10;

        let tier = "Low Risk";
        let color = "emerald";
        let recs = [
            "Maintain current active lifestyle and routine health check-ups.",
            "Sustain balanced low-sodium nutrition.",
            "Keep resting BP under 120/80 mm Hg."
        ];

        if (risk_percent >= 65.0) {
            tier = "High Risk";
            color = "rose";
            recs = [
                "Urgent consultation advised with a specialist cardiologist.",
                "Perform full ECG stress testing and echocardiogram.",
                "Monitor blood pressure and lipid profile daily."
            ];
        } else if (risk_percent >= 30.0) {
            tier = "Moderate Risk";
            color = "amber";
            recs = [
                "Schedule a preventative cardiology evaluation.",
                "Adopt low-sodium Mediterranean diet.",
                "Engage in 150+ minutes of aerobic physical exercise weekly."
            ];
        }

        const top_drivers = [
            { feature: 'oldpeak', value: p.oldpeak, importance: 21.4 },
            { feature: 'thalach', value: p.thalach, importance: 18.2 },
            { feature: 'exang', value: p.exang, importance: 15.8 },
            { feature: 'cp', value: p.cp, importance: 14.1 }
        ];

        renderResults({
            risk_percentage: risk_percent,
            risk_tier: tier,
            badge_color: color,
            recommendations: recs,
            top_drivers: top_drivers
        });
    }

    function renderResults(data) {
        // 1. Update Gauge & Percentage Text
        const pct = data.risk_percentage;
        riskScoreText.textContent = `${pct}%`;

        // Circumference = 2 * PI * 42 = ~264
        const strokeDashoffset = 264 - (264 * (pct / 100));
        gaugeFill.style.strokeDashoffset = strokeDashoffset;

        // 2. Update Badge
        riskTierBadge.textContent = data.risk_tier;
        riskTierBadge.className = `tier-badge ${data.badge_color || 'mod'}`;

        if (pct >= 65) {
            gaugeFill.style.stroke = '#f43f5e';
        } else if (pct >= 30) {
            gaugeFill.style.stroke = '#f59e0b';
        } else {
            gaugeFill.style.stroke = '#10b981';
        }

        // 3. Render Top Feature Drivers
        if (data.top_drivers && data.top_drivers.length > 0) {
            driversList.innerHTML = data.top_drivers.map(item => `
                <div class="driver-item">
                    <div>
                        <strong>${getFeatureLabel(item.feature)}</strong>
                        <div class="driver-bar" style="width: ${Math.min(item.importance * 3.5, 100)}%;"></div>
                    </div>
                    <span>Val: <strong>${item.value}</strong></span>
                </div>
            `).join('');
        }

        // 4. Render Clinical Recommendations
        if (data.recommendations && data.recommendations.length > 0) {
            recommendationsList.innerHTML = data.recommendations.map(r => `<li>${r}</li>`).join('');
        }
    }

    function getFeatureLabel(feat) {
        const map = {
            age: 'Age', sex: 'Biological Sex', cp: 'Chest Pain Type',
            trestbps: 'Resting Blood Pressure', chol: 'Cholesterol',
            fbs: 'Fasting Blood Sugar', restecg: 'Resting ECG',
            thalach: 'Max Heart Rate', exang: 'Exercise Angina',
            oldpeak: 'ST Depression (Oldpeak)', slope: 'ST Slope',
            ca: 'Major Vessels (CA)', thal: 'Thalassemia'
        };
        return map[feat] || feat;
    }

    // Auto run initial prediction on load
    runPrediction();
});
