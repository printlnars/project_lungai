document.addEventListener('DOMContentLoaded', () => {
  console.log('DOM loaded, initializing...');

  // ── ELEMENTS ──
  const form = document.getElementById('prediction-form');
  console.log('Form found:', !!form);

  if (!form) {
    console.error('Form not found!');
    return;
  }

  const navItems = document.querySelectorAll('.nav-item');
  const tabContents = document.querySelectorAll('.tab-content');
  const smokerSelect = document.getElementById('smoker');
  const smokingDetails = document.getElementById('smoking-details');
  const resultsWrap = document.getElementById('results-container');
  const loadingOverlay = document.getElementById('loading-overlay');
  const saveBtn = document.getElementById('save-patient-btn');
  const featureList = document.getElementById('feature-importance-list');
  const recList = document.getElementById('recommendations-list');
  const confidenceVal = document.getElementById('confidence-val');
  const confidenceBar = document.getElementById('confidence-bar');
  const similarCases = document.getElementById('similar-cases');
  const riskGroup = document.getElementById('risk-group');
  const gaugePct = document.getElementById('gauge-pct');
  const gaugeCircle = document.getElementById('gauge-circle');
  const riskBadge = document.getElementById('risk-badge');
  const riskText = document.getElementById('risk-text');
  const riskDesc = document.getElementById('risk-desc');
  const processLog = document.getElementById('process-log');

  let riskChart = null;
  let riskDistChart = null;
  let activityChart = null;
  let riskGroupChart = null;
  let lastData = null;

  const RADIUS = 75;
  const CIRCUM = 2 * Math.PI * RADIUS;

  // ── SIDEBAR NAV ──
  navItems.forEach(btn => {
    btn.addEventListener('click', () => {
      const tab = btn.dataset.tab;
      navItems.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));
      btn.classList.add('active');
      const targetTab = document.getElementById(`${tab}-tab`);
      if (targetTab) {
        targetTab.classList.add('active');
      }

      if (tab === 'analytics') {
        setTimeout(() => initAnalyticsCharts(), 100);
      }
    });
  });

  // ── SMOKING TOGGLE ──
  if (smokerSelect) {
    smokerSelect.addEventListener('change', e => {
      if (e.target.value === 'choice_1') {
        smokingDetails.style.display = 'block';
      } else {
        smokingDetails.style.display = 'none';
        document.getElementById('smoking_years').value = 'choice_0';
        document.getElementById('cigs_per_day').value = 'choice_0';
      }
    });
  }

  // ── FORM SUBMIT ──
  form.addEventListener('submit', async e => {
    e.preventDefault();
    console.log('Form submitted!');
    showLoading();

    const formData = new FormData(form);
    const data = {};
    formData.forEach((val, key) => {
      data[key] = (!isNaN(val) && val !== '') ? parseFloat(val) : val;
    });
    data['LUNG_CANCER'] = 0;

    console.log('Form data:', data);
    await runLoadingSteps();

    try {
      console.log('Sending fetch request...');
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
      });

      console.log('Response status:', res.status);
      const result = await res.json();
      console.log('Result:', result);
      hideLoading();

      if (result.status === 'success') {
        lastData = { ...data, LUNG_CANCER: result.prediction };
        showResults(result);
      } else {
        alert('Error: ' + (result.error || 'Unknown error'));
      }
    } catch (err) {
      hideLoading();
      console.error('Error:', err);
      alert('Error connecting to server: ' + err.message);
    }
  });

  // ── LOADING SIMULATION ──
  function showLoading() {
    loadingOverlay.classList.remove('hidden');
    processLog.innerHTML = '';
    [1, 2, 3, 4].forEach(i => {
      const step = document.getElementById(`step-${i}`);
      if (step) step.classList.remove('active');
    });
  }

  function hideLoading() {
    loadingOverlay.classList.add('hidden');
  }

  async function runLoadingSteps() {
    const logs = [
      '> Initializing diagnostic engine...',
      '> Loading patient vectors...',
      '> Mapping clinical constraints...',
      '> Running CatBoost inference...',
      '> Blending risk distributions...',
      '> Generating clinical report...',
    ];
    let logIdx = 0;
    const logInterval = setInterval(() => {
      if (logIdx < logs.length) {
        const d = document.createElement('div');
        d.textContent = logs[logIdx++];
        processLog.appendChild(d);
        processLog.scrollTop = processLog.scrollHeight;
      }
    }, 280);

    const steps = [1, 2, 3, 4];
    for (const s of steps) {
      const stepEl = document.getElementById(`step-${s}`);
      if (stepEl) stepEl.classList.add('active');
      await sleep(400);
    }
    await sleep(300);
    clearInterval(logInterval);
  }

  function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
  }

  // ── SHOW RESULTS ──
  function showResults(result) {
    const prob = result.probability;
    resultsWrap.classList.remove('hidden');
    resultsWrap.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // Gauge
    gaugePct.textContent = prob.toFixed(1) + '%';
    const offset = CIRCUM - (prob / 100) * CIRCUM;
    gaugeCircle.style.strokeDashoffset = offset;

    // Risk badge & color
    riskBadge.className = 'risk-badge';
    if (prob > 60) {
      riskBadge.classList.add('high');
      riskText.textContent = 'High Risk';
      gaugeCircle.style.stroke = '#ef4444';
      riskDesc.textContent = 'Significant risk factors identified. Urgent consultation with an oncologist and further examination are highly recommended.';
    } else if (prob > 30) {
      riskBadge.classList.add('mid');
      riskText.textContent = 'Moderate Risk';
      gaugeCircle.style.stroke = '#f59e0b';
      riskDesc.textContent = 'Moderate risk factors identified. Pulmonologist consultation and regular monitoring are recommended.';
    } else {
      riskBadge.className = 'risk-badge low';
      riskText.textContent = 'Low Risk';
      gaugeCircle.style.stroke = '#10b981';
      riskDesc.textContent = 'No significant risk factors identified. It is recommended to maintain a healthy lifestyle and undergo annual physicals.';
    }

    buildChart(prob);
    renderReport(result);
  }

  function renderReport(data) {
    if (!featureList) return;

    // Feature importance
    featureList.innerHTML = data.feature_importance.map(f => `
      <div class="feature-row">
        <div class="feature-info">
          <span class="feature-name">${f.name}</span>
          <span class="feature-val">${f.value}%</span>
        </div>
        <div class="feature-bar-bg">
          <div class="feature-bar-fill" style="width:${f.value}%"></div>
        </div>
      </div>
    `).join('');

    // Confidence
    if (confidenceVal && confidenceBar) {
      const conf = data.metadata.confidence;
      confidenceVal.textContent = conf;
      confidenceBar.style.width = conf + '%';
    }

    // Recommendations
    if (recList) {
      recList.innerHTML = data.recommendations.map(r => `<li>${r}</li>`).join('');
    }

    // Comparison
    if (similarCases && riskGroup) {
      similarCases.textContent = data.metadata.similar_cases;
      riskGroup.textContent = data.metadata.risk_group;
    }
  }

  function buildChart(prob) {
    const chartEl = document.getElementById('riskChart');
    if (!chartEl) return;

    const ctx = chartEl.getContext('2d');
    if (riskChart) riskChart.destroy();

    riskChart = new Chart(ctx, {
      type: 'doughnut',
      data: {
        labels: ['Lung Cancer Risk', 'Normal'],
        datasets: [{
          data: [prob, 100 - prob],
          backgroundColor: [
            prob > 60 ? '#ef4444' : prob > 30 ? '#f59e0b' : '#10b981',
            'rgba(255,255,255,0.06)'
          ],
          borderWidth: 0,
          hoverOffset: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: '72%',
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#94a3b8', font: { family: 'Inter', size: 13 }, padding: 16 }
          }
        }
      }
    });
  }

  // ── INITIALIZE ANALYTICS CHARTS ──
  function initAnalyticsCharts() {
    // Risk Distribution
    const ctxRiskDist = document.getElementById('riskDistChart');
    if (ctxRiskDist) {
      const riskDistCtx = ctxRiskDist.getContext('2d');
      if (riskDistChart) riskDistChart.destroy();
      riskDistChart = new Chart(riskDistCtx, {
        type: 'bar',
        data: {
          labels: ['<10%', '10-30%', '30-60%', '>60%'],
          datasets: [{
            label: 'Number of Patients',
            data: [2100, 1247, 650, 250],
            backgroundColor: ['#10b981', '#f59e0b', '#ef4444', '#dc2626'],
            borderRadius: 8,
            borderSkipped: false
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: '#94a3b8' } }
          },
          scales: {
            y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
          }
        }
      });
    }

    // Activity Chart
    const ctxActivity = document.getElementById('activityChart');
    if (ctxActivity) {
      const actCtx = ctxActivity.getContext('2d');
      if (activityChart) activityChart.destroy();
      activityChart = new Chart(actCtx, {
        type: 'line',
        data: {
          labels: ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00'],
          datasets: [{
            label: 'Diagnoses',
            data: [12, 25, 45, 89, 156, 112],
            borderColor: '#6366f1',
            backgroundColor: 'rgba(99,102,241,0.1)',
            fill: true,
            tension: 0.4,
            pointRadius: 4,
            pointBackgroundColor: '#6366f1'
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { labels: { color: '#94a3b8' } }
          },
          scales: {
            y: { ticks: { color: '#94a3b8' }, grid: { color: 'rgba(255,255,255,0.05)' } },
            x: { ticks: { color: '#94a3b8' }, grid: { display: false } }
          }
        }
      });
    }

    // Risk Group Chart
    const ctxRiskGroup = document.getElementById('riskGroupChart');
    if (ctxRiskGroup) {
      const riskGroupCtx = ctxRiskGroup.getContext('2d');
      if (riskGroupChart) riskGroupChart.destroy();
      riskGroupChart = new Chart(riskGroupCtx, {
        type: 'doughnut',
        data: {
          labels: ['G-I (Low)', 'G-II (Moderate)', 'G-III (Elevated)', 'G-IV (High)'],
          datasets: [{
            data: [2100, 1400, 650, 97],
            backgroundColor: ['#10b981', '#06b6d4', '#f59e0b', '#ef4444'],
            borderWidth: 0
          }]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: 'bottom', labels: { color: '#94a3b8' } }
          }
        }
      });
    }
  }

  // ── SAVE PATIENT ──
  if (saveBtn) {
    saveBtn.addEventListener('click', async () => {
      if (!lastData) return;
      try {
        const res = await fetch('/save_patient', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(lastData)
        });
        const result = await res.json();
        if (result.status === 'success') {
          saveBtn.innerHTML = '<i class="fas fa-check"></i> Saved';
          saveBtn.disabled = true;
          saveBtn.style.opacity = '0.6';
          setTimeout(() => {
            saveBtn.innerHTML = '<i class="fas fa-save"></i> Save';
            saveBtn.disabled = false;
            saveBtn.style.opacity = '1';
          }, 3000);
        }
      } catch (err) {
        console.error(err);
      }
    });
  }

  console.log('Initialization complete!');
});
