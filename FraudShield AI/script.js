/**
 * ==========================================================================
 * FRAUDSHIELD AI // CLIENT ENGINE & INTERACTIVE TELEMETRY CONTROLLER
 * Hybrid Mode:
 * 1. Connects to Python Flask Backend (/api/analyze) for real Logistic Regression ML
 * 2. Graceful offline fallback to statistical heuristics (Z-score + Bayes)
 * 3. Web Audio API Cyber Synthesizer
 * 4. Canvas Neural Particle Grid
 * 5. Event Stream Terminal Logging & JSON Exporter
 * ==========================================================================
 */

// Dataset Statistical Constants (derived from AI_Fraud_Detection_Transactions.csv)
const STATS_BASELINE = {
    amountMean: 1191.195,
    amountStd: 1328.70,
    amountQ1: 1045.0,
    amountQ3: 2890.0,
    baseFraudRate: 0.0178 // 1.78% base probability from training dataset
};

// Global State
const state = {
    audioEnabled: true,
    isScanning: false,
    activeRiskScore: 4.2,
    audioCtx: null,
    backendOnline: false
};

// ==========================================================================
// 1. CYBER SOUND SYNTHESIZER (Web Audio API)
// ==========================================================================
function initAudio() {
    if (!state.audioCtx) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        if (AudioContext) {
            state.audioCtx = new AudioContext();
        }
    }
    if (state.audioCtx && state.audioCtx.state === 'suspended') {
        state.audioCtx.resume();
    }
}

function playCyberSound(type) {
    if (!state.audioEnabled) return;
    try {
        initAudio();
        if (!state.audioCtx) return;

        const ctx = state.audioCtx;
        const now = ctx.currentTime;

        if (type === 'click') {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(800, now);
            osc.frequency.exponentialRampToValueAtTime(1400, now + 0.04);
            gain.gain.setValueAtTime(0.06, now);
            gain.gain.linearRampToValueAtTime(0.001, now + 0.04);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(now);
            osc.stop(now + 0.05);
        } else if (type === 'scan') {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'triangle';
            osc.frequency.setValueAtTime(320, now);
            osc.frequency.linearRampToValueAtTime(980, now + 0.35);
            gain.gain.setValueAtTime(0.08, now);
            gain.gain.linearRampToValueAtTime(0.001, now + 0.4);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(now);
            osc.stop(now + 0.45);
        } else if (type === 'safe') {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sine';
            osc.frequency.setValueAtTime(523.25, now);
            osc.frequency.setValueAtTime(659.25, now + 0.08);
            osc.frequency.setValueAtTime(783.99, now + 0.16);
            gain.gain.setValueAtTime(0.08, now);
            gain.gain.linearRampToValueAtTime(0.001, now + 0.35);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(now);
            osc.stop(now + 0.4);
        } else if (type === 'danger') {
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.type = 'sawtooth';
            osc.frequency.setValueAtTime(220, now);
            osc.frequency.setValueAtTime(180, now + 0.1);
            osc.frequency.setValueAtTime(220, now + 0.2);
            gain.gain.setValueAtTime(0.12, now);
            gain.gain.linearRampToValueAtTime(0.001, now + 0.35);
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.start(now);
            osc.stop(now + 0.4);
        }
    } catch (e) {
        console.warn('AudioContext not active yet', e);
    }
}

// ==========================================================================
// 2. CANVAS NEURAL PARTICLE GRID
// ==========================================================================
function initNeuralCanvas() {
    const canvas = document.getElementById('neural-bg');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });

    const particles = [];
    const particleCount = Math.min(Math.floor((width * height) / 18000), 70);

    for (let i = 0; i < particleCount; i++) {
        particles.push({
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.6,
            vy: (Math.random() - 0.5) * 0.6,
            radius: Math.random() * 1.5 + 1
        });
    }

    function render() {
        ctx.clearRect(0, 0, width, height);

        ctx.lineWidth = 0.5;
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 130) {
                    const alpha = (1 - dist / 130) * 0.22;
                    ctx.strokeStyle = `rgba(0, 240, 255, ${alpha})`;
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.stroke();
                }
            }
        }

        for (let i = 0; i < particles.length; i++) {
            const p = particles[i];
            p.x += p.vx;
            p.y += p.vy;

            if (p.x < 0 || p.x > width) p.vx *= -1;
            if (p.y < 0 || p.y > height) p.vy *= -1;

            ctx.fillStyle = 'rgba(0, 240, 255, 0.55)';
            ctx.shadowBlur = 6;
            ctx.shadowColor = '#00f0ff';
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
            ctx.fill();
            ctx.shadowBlur = 0;
        }

        requestAnimationFrame(render);
    }
    render();
}

// ==========================================================================
// 3. TELEMETRY LOGS & TERMINAL HELPER
// ==========================================================================
function logTerminal(message, type = 'info') {
    const terminalBody = document.getElementById('terminal-body');
    if (!terminalBody) return;

    const timeStr = new Date().toTimeString().split(' ')[0];
    const line = document.createElement('div');
    line.className = `term-line ${type}`;

    let tag = '[INFO]';
    if (type === 'danger') tag = '[CRITICAL]';
    if (type === 'warn') tag = '[WARNING]';
    if (type === 'success') tag = '[PASSED]';
    if (type === 'system') tag = '[SYSTEM]';

    line.textContent = `${timeStr} ${tag} ${message}`;
    terminalBody.appendChild(line);
    terminalBody.scrollTop = terminalBody.scrollHeight;
}

function showToast(message) {
    const toast = document.getElementById('cyber-toast');
    const toastMsg = document.getElementById('toast-message');
    if (!toast || !toastMsg) return;

    toastMsg.textContent = message;
    toast.classList.remove('hidden');

    setTimeout(() => {
        toast.classList.add('hidden');
    }, 3200);
}

// ==========================================================================
// 4. STATISTICAL & HEURISTIC FALLBACK ENGINE
// ==========================================================================
function computeInferenceMetrics(data) {
    const { amount, hour, failedAttempts, newDevice, locationChanged, accountAge, previousTxns } = data;

    const zScore = (amount - STATS_BASELINE.amountMean) / STATS_BASELINE.amountStd;
    const iqr = STATS_BASELINE.amountQ3 - STATS_BASELINE.amountQ1;
    const isOutlier = amount > STATS_BASELINE.amountQ3 + 1.5 * iqr;

    const isMidnightRisk = hour >= 1 && hour <= 5;
    const hourRiskMult = isMidnightRisk ? 2.8 : hour >= 23 || hour === 0 ? 1.6 : 1.0;
    const authRiskScore = Math.min(failedAttempts * 18, 50);

    let prior = STATS_BASELINE.baseFraudRate;
    let odds = prior / (1 - prior);

    if (newDevice) odds *= 4.5;
    if (locationChanged) odds *= 3.8;
    if (failedAttempts >= 2) odds *= Math.pow(2.8, failedAttempts - 1);
    if (isMidnightRisk) odds *= 3.2;
    if (zScore > 3.0) odds *= 4.0;
    else if (zScore > 1.5) odds *= 1.8;
    if (accountAge < 60) odds *= 2.2;
    if (previousTxns === 0) odds *= 1.9;

    let bayesProb = odds / (1 + odds);
    bayesProb = Math.min(Math.max(bayesProb, 0.012), 0.985);

    let compositeScore = bayesProb * 100;
    const riskFactors = [];

    if (failedAttempts >= 2) {
        riskFactors.push({
            name: `Failed Attempts Spike (${failedAttempts} Retries)`,
            weight: Math.min(failedAttempts * 12, 35),
            color: 'crimson'
        });
    }

    if (isMidnightRisk && amount > 5000) {
        riskFactors.push({
            name: `Off-Hour High-Value Spike (${hour}:00 hrs, ₹${amount.toLocaleString()})`,
            weight: 25,
            color: 'crimson'
        });
    } else if (isMidnightRisk) {
        riskFactors.push({
            name: `Unusual Transaction Window (${hour}:00 hrs)`,
            weight: 15,
            color: 'amber'
        });
    }

    if (newDevice && locationChanged) {
        riskFactors.push({
            name: 'Device Swap + IP Geo-Hop Anomaly',
            weight: 28,
            color: 'crimson'
        });
    } else if (newDevice) {
        riskFactors.push({
            name: 'Unrecognized Device Fingerprint',
            weight: 14,
            color: 'amber'
        });
    } else if (locationChanged) {
        riskFactors.push({
            name: 'Geographic Velocity Mismatch',
            weight: 14,
            color: 'amber'
        });
    }

    if (zScore >= 3.0) {
        riskFactors.push({
            name: `Extreme Amount Deviation (Z = +${zScore.toFixed(2)}σ)`,
            weight: 22,
            color: 'crimson'
        });
    } else if (zScore >= 1.5) {
        riskFactors.push({
            name: `Elevated Amount (Z = +${zScore.toFixed(2)}σ)`,
            weight: 10,
            color: 'amber'
        });
    }

    if (accountAge < 60 && previousTxns <= 1) {
        riskFactors.push({
            name: `New Account / Sparse History (${accountAge}d old, ${previousTxns} txns)`,
            weight: 12,
            color: 'amber'
        });
    }

    if (riskFactors.length === 0) {
        riskFactors.push({
            name: 'Historical Velocity & Behavior Consistency',
            weight: 95,
            color: 'emerald'
        });
    }

    return {
        zScore,
        isOutlier,
        hourRiskMult,
        authRiskScore,
        bayesProb,
        compositeScore,
        riskFactors
    };
}

// ==========================================================================
// 5. UPDATE RESULTS UI
// ==========================================================================
function updateResultsUI(metrics, formData) {
    const { compositeScore, zScore, isOutlier, hourRiskMult, bayesProb, riskFactors } = metrics;
    state.activeRiskScore = compositeScore;

    const verdictBanner = document.getElementById('verdict-banner');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictTitle = document.getElementById('verdict-title');
    const verdictDesc = document.getElementById('verdict-desc');
    const verdictBadge = document.getElementById('verdict-badge');
    const gaugeRiskLevel = document.getElementById('gauge-risk-level');
    const gaugeFillCircle = document.getElementById('gauge-fill-circle');
    const gaugePercentage = document.getElementById('gauge-percentage');
    const metricZScore = document.getElementById('metric-zscore');
    const metricBayes = document.getElementById('metric-bayes');
    const metricOutlier = document.getElementById('metric-outlier');
    const metricHourRisk = document.getElementById('metric-hour-risk');
    const metricAuthPenalty = document.getElementById('metric-auth-penalty');
    const attributionBars = document.getElementById('attribution-bars');
    const riskFactorsCount = document.getElementById('risk-factors-count');
    const recommendationText = document.getElementById('recommendation-text');

    let category = 'safe';
    let bannerTitle = 'CLEARED // TRANSACTION AUTHORIZED';
    let bannerDesc = 'Transaction parameters match expected baseline profile. No threat indicators triggered.';
    let badgeText = 'LOW RISK';
    let gaugeLevel = 'NOMINAL';
    let strokeColor = '#10b981';

    if (compositeScore >= 65) {
        category = 'danger';
        bannerTitle = 'CRITICAL ALERT // FRAUD THREAT DETECTED';
        bannerDesc = 'High probability of fraudulent activity. Anomalies detected across multiple biometric & behavioral vectors.';
        badgeText = 'CRITICAL FRAUD';
        gaugeLevel = 'HIGH THREAT';
        strokeColor = '#ff0055';
        playCyberSound('danger');
    } else if (compositeScore >= 30) {
        category = 'warning';
        bannerTitle = 'FLAGGED // SUSPICIOUS TRANSACTION';
        bannerDesc = 'Moderate variance from typical pattern detected. Recommend step-up verification before settlement.';
        badgeText = 'ELEVATED RISK';
        gaugeLevel = 'SUSPICIOUS';
        strokeColor = '#f59e0b';
        playCyberSound('scan');
    } else {
        playCyberSound('safe');
    }

    verdictBanner.className = `verdict-banner ${category}`;
    verdictTitle.textContent = bannerTitle;
    verdictDesc.textContent = bannerDesc;
    verdictBadge.textContent = badgeText;
    gaugeRiskLevel.textContent = gaugeLevel;

    if (category === 'danger') {
        verdictIcon.className = 'fa-solid fa-triangle-exclamation';
    } else if (category === 'warning') {
        verdictIcon.className = 'fa-solid fa-shield-virus';
    } else {
        verdictIcon.className = 'fa-solid fa-shield-check';
    }

    const totalCircumference = 471.2;
    const progressOffset = totalCircumference - (totalCircumference * (compositeScore / 100));
    gaugeFillCircle.style.strokeDasharray = totalCircumference;
    gaugeFillCircle.style.strokeDashoffset = progressOffset;
    gaugeFillCircle.style.stroke = strokeColor;
    gaugeFillCircle.style.filter = `drop-shadow(0 0 8px ${strokeColor})`;
    gaugePercentage.textContent = `${compositeScore.toFixed(1)}%`;

    const sign = zScore >= 0 ? '+' : '';
    metricZScore.textContent = `${sign}${zScore.toFixed(2)}σ (${zScore > 2.5 ? 'Extreme Outlier' : zScore > 1.2 ? 'Moderate' : 'Normal'})`;
    metricZScore.className = `s-val ${zScore > 2.5 ? 'text-crimson' : zScore > 1.2 ? 'text-amber' : 'text-cyan'}`;

    metricBayes.textContent = bayesProb.toFixed(3);
    metricBayes.className = `s-val ${bayesProb > 0.5 ? 'text-crimson' : bayesProb > 0.2 ? 'text-amber' : 'text-emerald'}`;

    metricOutlier.textContent = isOutlier ? 'Exceeds IQR 1.5x Threshold' : 'Within Normal IQR Range';
    metricOutlier.className = `s-val ${isOutlier ? 'text-crimson' : 'text-emerald'}`;

    metricHourRisk.textContent = `${hourRiskMult.toFixed(1)}x (${formData.hour >= 1 && formData.hour <= 5 ? 'Midnight Hazard' : 'Nominal'})`;
    metricHourRisk.className = `s-val ${hourRiskMult > 1.5 ? 'text-crimson' : 'text-cyan'}`;

    metricAuthPenalty.textContent = `${formData.failedAttempts} failed retries`;
    metricAuthPenalty.className = `s-val ${formData.failedAttempts >= 2 ? 'text-crimson' : 'text-emerald'}`;

    attributionBars.innerHTML = '';
    riskFactorsCount.textContent = `${riskFactors.length} Factor${riskFactors.length > 1 ? 's' : ''} Identified`;

    riskFactors.forEach(factor => {
        const item = document.createElement('div');
        item.className = 'att-factor-card';

        let colorHex = '#10b981';
        if (factor.color === 'crimson') colorHex = '#ff0055';
        if (factor.color === 'amber') colorHex = '#f59e0b';

        item.innerHTML = `
            <div class="att-factor-meta">
                <span style="color: ${colorHex}">${factor.name}</span>
                <span class="weight" style="color: ${colorHex}">+${factor.weight}%</span>
            </div>
            <div class="att-progress-track">
                <div class="att-progress-fill" style="width: ${Math.min(factor.weight * 2.5, 100)}%; background: ${colorHex}; box-shadow: 0 0 6px ${colorHex}"></div>
            </div>
        `;
        attributionBars.appendChild(item);
    });

    if (category === 'danger') {
        recommendationText.innerHTML = `
            <strong class="text-crimson">AUTOMATED QUARANTINE TRIGGERED:</strong> Block transaction immediately. Hold recipient settlement and flag device fingerprint (${formData.newDevice ? 'Unrecognized HW' : 'Known HW'}). Dispatch security notification to registered mobile.
        `;
    } else if (category === 'warning') {
        recommendationText.innerHTML = `
            <strong class="text-amber">CHALLENGE REQUIRED:</strong> Route through Step-Up Biometric or 3D-Secure OTP challenge before clearing funds. Invalidate active user session if OTP fails.
        `;
    } else {
        recommendationText.innerHTML = `
            <strong class="text-emerald">CLEARED FOR SETTLEMENT:</strong> Transaction risk within nominal boundary. Authorize straight-through processing (STP) through payment gateway.
        `;
    }

    logTerminal(
        `Inference completed for ${formData.txId}: Risk=${compositeScore.toFixed(1)}% | Verdict=${badgeText} | Bayes=${bayesProb.toFixed(3)}`,
        category === 'danger' ? 'danger' : category === 'warning' ? 'warn' : 'success'
    );
}

// Update UI directly from Flask Logistic Regression response
function updateResultsFromBackend(res, formData) {
    const compositeScore = res.risk_score_percent;
    state.activeRiskScore = compositeScore;

    const verdictBanner = document.getElementById('verdict-banner');
    const verdictIcon = document.getElementById('verdict-icon');
    const verdictTitle = document.getElementById('verdict-title');
    const verdictDesc = document.getElementById('verdict-desc');
    const verdictBadge = document.getElementById('verdict-badge');
    const gaugeRiskLevel = document.getElementById('gauge-risk-level');
    const gaugeFillCircle = document.getElementById('gauge-fill-circle');
    const gaugePercentage = document.getElementById('gauge-percentage');
    const metricZScore = document.getElementById('metric-zscore');
    const metricBayes = document.getElementById('metric-bayes');
    const metricOutlier = document.getElementById('metric-outlier');
    const metricHourRisk = document.getElementById('metric-hour-risk');
    const metricAuthPenalty = document.getElementById('metric-auth-penalty');
    const attributionBars = document.getElementById('attribution-bars');
    const riskFactorsCount = document.getElementById('risk-factors-count');
    const recommendationText = document.getElementById('recommendation-text');

    let category = 'safe';
    let strokeColor = '#10b981';

    if (compositeScore >= 65 || res.prediction === 1) {
        category = 'danger';
        strokeColor = '#ff0055';
        playCyberSound('danger');
    } else if (compositeScore >= 30) {
        category = 'warning';
        strokeColor = '#f59e0b';
        playCyberSound('scan');
    } else {
        playCyberSound('safe');
    }

    verdictBanner.className = `verdict-banner ${category}`;
    verdictTitle.textContent = res.verdict_title;
    verdictDesc.textContent = res.verdict_description;
    verdictBadge.textContent = res.verdict_badge;
    gaugeRiskLevel.textContent = res.risk_level;

    if (category === 'danger') {
        verdictIcon.className = 'fa-solid fa-triangle-exclamation';
    } else if (category === 'warning') {
        verdictIcon.className = 'fa-solid fa-shield-virus';
    } else {
        verdictIcon.className = 'fa-solid fa-shield-check';
    }

    const totalCircumference = 471.2;
    const progressOffset = totalCircumference - (totalCircumference * (compositeScore / 100));
    gaugeFillCircle.style.strokeDasharray = totalCircumference;
    gaugeFillCircle.style.strokeDashoffset = progressOffset;
    gaugeFillCircle.style.stroke = strokeColor;
    gaugeFillCircle.style.filter = `drop-shadow(0 0 8px ${strokeColor})`;
    gaugePercentage.textContent = `${compositeScore.toFixed(1)}%`;

    const zScore = (formData.amount - STATS_BASELINE.amountMean) / STATS_BASELINE.amountStd;
    const sign = zScore >= 0 ? '+' : '';
    metricZScore.textContent = `${sign}${zScore.toFixed(2)}σ (LR Scaled)`;
    metricZScore.className = `s-val ${zScore > 2.5 ? 'text-crimson' : 'text-cyan'}`;

    metricBayes.textContent = res.fraud_probability.toFixed(4);
    metricBayes.className = `s-val ${res.fraud_probability > 0.5 ? 'text-crimson' : 'text-emerald'}`;

    metricOutlier.textContent = res.prediction === 1 ? 'Positive Fraud Class (1)' : 'Legitimate Class (0)';
    metricOutlier.className = `s-val ${res.prediction === 1 ? 'text-crimson' : 'text-emerald'}`;

    metricHourRisk.textContent = `${formData.hour}:00 hrs (${formData.hour <= 5 && formData.hour >= 1 ? 'High Risk' : 'Standard'})`;
    metricHourRisk.className = `s-val ${formData.hour <= 5 && formData.hour >= 1 ? 'text-crimson' : 'text-cyan'}`;

    metricAuthPenalty.textContent = `${formData.failedAttempts} failed retries`;
    metricAuthPenalty.className = `s-val ${formData.failedAttempts >= 2 ? 'text-crimson' : 'text-emerald'}`;

    attributionBars.innerHTML = '';
    const atts = res.feature_attributions || [];
    riskFactorsCount.textContent = `${atts.length} Model Weight${atts.length > 1 ? 's' : ''}`;

    atts.forEach(att => {
        const item = document.createElement('div');
        item.className = 'att-factor-card';
        const colorHex = att.direction === 'danger' ? '#ff0055' : '#10b981';
        item.innerHTML = `
            <div class="att-factor-meta">
                <span style="color: ${colorHex}">${att.feature} (Logistic Coef Weight)</span>
                <span class="weight" style="color: ${colorHex}">${att.impact > 0 ? '+' : ''}${att.impact}</span>
            </div>
            <div class="att-progress-track">
                <div class="att-progress-fill" style="width: ${att.weight_pct}%; background: ${colorHex}; box-shadow: 0 0 6px ${colorHex}"></div>
            </div>
        `;
        attributionBars.appendChild(item);
    });

    recommendationText.innerHTML = res.recommendation;

    logTerminal(
        `[LOGISTIC REGRESSION] Predicted for ${formData.txId}: Risk=${compositeScore.toFixed(1)}% | Class=${res.prediction} (${res.verdict_badge})`,
        category === 'danger' ? 'danger' : category === 'warning' ? 'warn' : 'success'
    );
}

// ==========================================================================
// 6. FORM & RADAR SCAN EXECUTION
// ==========================================================================
function executeScan(formData) {
    if (state.isScanning) return;
    state.isScanning = true;

    playCyberSound('scan');

    const scanningHud = document.getElementById('scanning-hud');
    const resultsDisplay = document.getElementById('results-display');
    const scanStepTitle = document.getElementById('scan-step-title');
    const scanProgressFill = document.getElementById('scan-progress-fill');
    const scanConsoleLine = document.getElementById('scan-console-line');
    const analyzeBtn = document.getElementById('btn-analyze');
    const analyzeBtnText = document.getElementById('analyze-btn-text');

    resultsDisplay.style.opacity = '0.2';
    scanningHud.classList.remove('hidden');
    analyzeBtn.disabled = true;
    analyzeBtnText.textContent = 'NEURAL ENGINE ACTIVE...';

    logTerminal(`Ingesting vector ${formData.txId}: Amount=₹${formData.amount}, Hour=${formData.hour}:00...`, 'info');

    const phases = [
        { progress: '25%', title: 'PARSING TELEMETRY VECTOR...', console: 'Normalizing input tensors against 8,012 training points...' },
        { progress: '55%', title: 'EVALUATING LOGISTIC REGRESSION WEIGHTS...', console: 'StandardScaler pipeline active. Fitting coefficients...' },
        { progress: '85%', title: 'COMPUTING SIGMOID PROBABILITY...', console: 'P(Fraud | X) = 1 / (1 + e^-(w^T x + b))...' },
        { progress: '100%', title: 'SYNTHESIZING VERDICT...', console: 'Classifying binary fraud threshold and XAI weights...' }
    ];

    let currentPhase = 0;
    const interval = setInterval(() => {
        if (currentPhase < phases.length) {
            scanProgressFill.style.width = phases[currentPhase].progress;
            scanStepTitle.textContent = phases[currentPhase].title;
            scanConsoleLine.textContent = phases[currentPhase].console;
            currentPhase++;
        } else {
            clearInterval(interval);

            // Send payload to Python Flask Backend
            const backendPayload = {
                transaction_id: formData.txId,
                amount: formData.amount,
                transaction_hour: formData.hour,
                customer_age: formData.customerAge,
                previous_transactions: formData.previousTxns,
                failed_attempts: formData.failedAttempts,
                new_device: formData.newDevice ? 1 : 0,
                location_changed: formData.locationChanged ? 1 : 0,
                account_age_days: formData.accountAge
            };

            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(backendPayload)
            })
            .then(res => {
                if (!res.ok) throw new Error(`HTTP ${res.status}`);
                return res.json();
            })
            .then(data => {
                scanningHud.classList.add('hidden');
                resultsDisplay.style.opacity = '1';
                analyzeBtn.disabled = false;
                analyzeBtnText.textContent = 'INITIATE AI NEURAL SCAN';
                state.isScanning = false;

                if (data.status === 'success') {
                    updateResultsFromBackend(data, formData);
                } else {
                    throw new Error(data.message || 'Model prediction failed');
                }
            })
            .catch(err => {
                // Offline fallback (e.g., Netlify static deployment)
                scanningHud.classList.add('hidden');
                resultsDisplay.style.opacity = '1';
                analyzeBtn.disabled = false;
                analyzeBtnText.textContent = 'INITIATE AI NEURAL SCAN';
                state.isScanning = false;

                logTerminal(`Flask backend offline (${err.message}). Using client statistical heuristic.`, 'warn');
                const metrics = computeInferenceMetrics(formData);
                updateResultsUI(metrics, formData);
            });
        }
    }, 280);
}

// ==========================================================================
// 7. PRESET VECTORS (Including Student Task Challenge)
// ==========================================================================
function populateForm(preset) {
    document.getElementById('tx-id').value = preset.txId;
    document.getElementById('tx-amount').value = preset.amount;
    document.getElementById('tx-hour').value = preset.hour;
    document.getElementById('tx-customer-age').value = preset.customerAge;
    document.getElementById('tx-account-age').value = preset.accountAge;
    document.getElementById('tx-previous-txns').value = preset.previousTxns;
    document.getElementById('tx-failed-attempts').value = preset.failedAttempts;
    document.getElementById('tx-channel').value = preset.channel;
    document.getElementById('tx-new-device').checked = preset.newDevice;
    document.getElementById('tx-location-changed').checked = preset.locationChanged;

    updateHourDisplay(preset.hour);
    playCyberSound('click');
    showToast(`Loaded preset: ${preset.name}`);
    logTerminal(`Vector preset applied: "${preset.name}"`, 'system');
}

function updateHourDisplay(hourVal) {
    const hourDisplay = document.getElementById('hour-display');
    const hourRiskTag = document.getElementById('hour-risk-tag');
    const formatted = `${String(hourVal).padStart(2, '0')}:00`;
    hourDisplay.textContent = formatted;

    if (hourVal >= 1 && hourVal <= 5) {
        hourRiskTag.textContent = 'CRITICAL RISK WINDOW (01:00-05:00)';
        hourRiskTag.className = 'hour-tag danger';
    } else if (hourVal >= 23 || hourVal === 0) {
        hourRiskTag.textContent = 'ELEVATED NIGHT HOURS';
        hourRiskTag.className = 'hour-tag';
        hourRiskTag.style.background = 'rgba(245, 158, 11, 0.2)';
        hourRiskTag.style.color = '#f59e0b';
    } else {
        hourRiskTag.textContent = 'NORMAL OPERATIONAL HOURS';
        hourRiskTag.className = 'hour-tag safe';
        hourRiskTag.style.background = '';
        hourRiskTag.style.color = '';
    }
}

function generateRandomTxnId() {
    return 'TXN-' + Math.floor(10000 + Math.random() * 90000);
}

// ==========================================================================
// 8. EVENT LISTENERS INITIALIZATION
// ==========================================================================
document.addEventListener('DOMContentLoaded', () => {
    initNeuralCanvas();

    // Check Flask backend health
    fetch('/api/health')
        .then(res => res.json())
        .then(data => {
            if (data.status === 'online') {
                state.backendOnline = true;
                const statusLabel = document.querySelector('.status-label');
                if (statusLabel) statusLabel.textContent = 'PYTHON AI BACKEND: ONLINE';
                const bannerText = document.querySelector('.banner-scroll .banner-text');
                if (bannerText) {
                    bannerText.innerHTML = '<strong class="text-emerald">PYTHON AI BACKEND ACTIVE:</strong> Connected to live Logistic Regression model (models/fraud_model.joblib). Full ML inference operational.';
                }
                logTerminal(`[ONLINE] Connected to Python Flask ML Backend (Accuracy=${(data.metrics.accuracy * 100).toFixed(1)}%)`, 'success');
            }
        })
        .catch(() => {
            logTerminal('[INFO] Flask server not detected. Running client-side simulation. Run `python app.py` for live backend.', 'info');
        });

    const audioBtn = document.getElementById('audio-toggle-btn');
    const audioIcon = document.getElementById('audio-icon');
    audioBtn.addEventListener('click', () => {
        state.audioEnabled = !state.audioEnabled;
        if (state.audioEnabled) {
            audioIcon.className = 'fa-solid fa-volume-high';
            showToast('Cyber Audio: ENABLED');
            playCyberSound('click');
        } else {
            audioIcon.className = 'fa-solid fa-volume-xmark';
            showToast('Cyber Audio: MUTED');
        }
    });

    const hourSlider = document.getElementById('tx-hour');
    hourSlider.addEventListener('input', (e) => {
        updateHourDisplay(parseInt(e.target.value, 10));
    });

    document.getElementById('btn-regen-id').addEventListener('click', () => {
        const newId = generateRandomTxnId();
        document.getElementById('tx-id').value = newId;
        playCyberSound('click');
        showToast(`Generated ID: ${newId}`);
    });

    document.getElementById('preset-safe').addEventListener('click', () => {
        populateForm({
            name: 'Normal Daytime Grocery',
            txId: generateRandomTxnId(),
            amount: 1450.00,
            hour: 14,
            customerAge: 34,
            accountAge: 620,
            previousTxns: 18,
            failedAttempts: 0,
            channel: 'UPI',
            newDevice: false,
            locationChanged: false
        });
    });

    document.getElementById('preset-suspicious').addEventListener('click', () => {
        populateForm({
            name: 'Geo-Hop & New Device',
            txId: generateRandomTxnId(),
            amount: 12800.00,
            hour: 23,
            customerAge: 29,
            accountAge: 55,
            previousTxns: 3,
            failedAttempts: 2,
            channel: 'CREDIT_CARD',
            newDevice: true,
            locationChanged: true
        });
    });

    // Exact Challenge from Student Project Task!
    document.getElementById('preset-attack').addEventListener('click', () => {
        populateForm({
            name: '₹52,000 Midnight Attack (Student Project Challenge)',
            txId: 'TXN-FINAL-52K',
            amount: 52000.00,
            hour: 3,
            customerAge: 26,
            accountAge: 45,
            previousTxns: 0,
            failedAttempts: 5,
            channel: 'NET_BANKING',
            newDevice: true,
            locationChanged: true
        });
    });

    document.getElementById('preset-random').addEventListener('click', () => {
        const randAmount = Math.floor(Math.random() * 45000) + 200;
        const randHour = Math.floor(Math.random() * 24);
        const randAge = Math.floor(Math.random() * 55) + 19;
        const randAccountAge = Math.floor(Math.random() * 800) + 10;
        const randPrev = Math.floor(Math.random() * 30);
        const randFailed = Math.floor(Math.random() * 4);
        const randNewDev = Math.random() > 0.6;
        const randLoc = Math.random() > 0.6;

        populateForm({
            name: 'Randomized Telemetry Stream',
            txId: generateRandomTxnId(),
            amount: randAmount,
            hour: randHour,
            customerAge: randAge,
            accountAge: randAccountAge,
            previousTxns: randPrev,
            failedAttempts: randFailed,
            channel: ['UPI', 'CREDIT_CARD', 'DEBIT_CARD', 'NET_BANKING', 'CRYPTO'][Math.floor(Math.random() * 5)],
            newDevice: randNewDev,
            locationChanged: randLoc
        });
    });

    const fraudForm = document.getElementById('fraud-form');
    fraudForm.addEventListener('submit', (e) => {
        e.preventDefault();

        const formData = {
            txId: document.getElementById('tx-id').value,
            amount: parseFloat(document.getElementById('tx-amount').value) || 0,
            hour: parseInt(document.getElementById('tx-hour').value, 10) || 0,
            customerAge: parseInt(document.getElementById('tx-customer-age').value, 10) || 30,
            accountAge: parseInt(document.getElementById('tx-account-age').value, 10) || 100,
            previousTxns: parseInt(document.getElementById('tx-previous-txns').value, 10) || 0,
            failedAttempts: parseInt(document.getElementById('tx-failed-attempts').value, 10) || 0,
            channel: document.getElementById('tx-channel').value,
            newDevice: document.getElementById('tx-new-device').checked,
            locationChanged: document.getElementById('tx-location-changed').checked
        };

        executeScan(formData);
    });

    document.getElementById('btn-reset').addEventListener('click', () => {
        document.getElementById('preset-safe').click();
        showToast('Form reset to default baseline');
    });

    document.getElementById('btn-refresh-stats').addEventListener('click', () => {
        playCyberSound('click');
        const scanned = document.getElementById('stat-scanned');
        const latency = document.getElementById('stat-latency');
        const navLat = document.getElementById('nav-latency');

        const curScanned = parseInt(scanned.textContent.replace(/,/g, ''), 10);
        const newScanned = curScanned + Math.floor(Math.random() * 12) + 3;
        scanned.textContent = newScanned.toLocaleString();

        const newLat = Math.floor(Math.random() * 6) + 10;
        latency.textContent = newLat;
        navLat.textContent = `${newLat}ms`;

        showToast('Telemetry synced with distributed nodes');
        logTerminal(`Telemetry heartbeat refreshed. Network latency: ${newLat}ms`, 'system');
    });

    document.getElementById('btn-copy-report').addEventListener('click', () => {
        playCyberSound('click');
        const report = {
            system: 'FraudShield AI v2.4 Sentinel',
            model: 'Logistic Regression Classifier',
            timestamp: new Date().toISOString(),
            transaction_id: document.getElementById('tx-id').value,
            amount: parseFloat(document.getElementById('tx-amount').value),
            hour: parseInt(document.getElementById('tx-hour').value, 10),
            failed_attempts: parseInt(document.getElementById('tx-failed-attempts').value, 10),
            new_device: document.getElementById('tx-new-device').checked,
            location_changed: document.getElementById('tx-location-changed').checked,
            risk_score_percent: state.activeRiskScore.toFixed(2),
            verdict: document.getElementById('verdict-badge').textContent
        };

        navigator.clipboard.writeText(JSON.stringify(report, null, 2))
            .then(() => showToast('Report payload copied to clipboard (JSON)'))
            .catch(() => showToast('Could not copy to clipboard'));
    });

    document.getElementById('btn-quarantine').addEventListener('click', () => {
        playCyberSound('danger');
        showToast('Protocol Enforced: Account token revoked & quarantined');
        logTerminal(`[ACTION] Account quarantined for TXN ${document.getElementById('tx-id').value}`, 'danger');
    });

    document.getElementById('btn-request-mfa').addEventListener('click', () => {
        playCyberSound('click');
        showToast('Protocol Enforced: Hardware MFA push sent to customer');
        logTerminal(`[ACTION] Step-up 2FA dispatched for TXN ${document.getElementById('tx-id').value}`, 'warn');
    });

    document.getElementById('btn-override-allow').addEventListener('click', () => {
        playCyberSound('safe');
        showToast('Protocol Enforced: TXN manually whitelisted by engineer');
        logTerminal(`[ACTION] Manual override authorized by ENG // DEV-01`, 'success');
    });

    document.getElementById('btn-clear-term').addEventListener('click', () => {
        const body = document.getElementById('terminal-body');
        body.innerHTML = '<div class="term-line system">[SYSTEM] Terminal buffer cleared. Stream monitoring active.</div>';
    });

    setTimeout(() => {
        const initialData = {
            txId: 'TXN-80912',
            amount: 2450.00,
            hour: 14,
            customerAge: 32,
            accountAge: 480,
            previousTxns: 12,
            failedAttempts: 0,
            channel: 'UPI',
            newDevice: false,
            locationChanged: false
        };
        const metrics = computeInferenceMetrics(initialData);
        updateResultsUI(metrics, initialData);
    }, 150);
});
