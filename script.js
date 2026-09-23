/* =========================================================
   Fintech Risk Lab — script.js
   1. Config & data          5. Output animators (data-anim)
   2. Cell code              6. runCell(cellId) + controls
   3. Output templates       7. Editor setup & boot
   4. Explanation cards
   ========================================================= */

/* ---------------------------------------------------------
   1. Config & data
   --------------------------------------------------------- */
const CANCEL = Symbol('cancel');
const REDUCE_MOTION = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const wait = ms => new Promise(resolve => setTimeout(resolve, REDUCE_MOTION ? Math.min(ms, 15) : ms));
const nextFrame = () => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));

// Numbers shown in the outputs. Edit here and every cell updates.
const METRICS = { accuracy: 88.5, precision: 86.2, recall: 84.8, f1: 85.5 };   // percent
const CM = { tn: 118, fp: 12, fn: 11, tp: 59 };
const RISK = 84.0;

const HEAD = ['credit_score', 'annual_income', 'loan_amount', 'debt_to_income_ratio', 'employment_status', 'home_ownership', 'loan_status'];
const ROWS = [
  [712,  7013500, 1826000, '0.213', 'Employed',      'Mortgage', 0],
  [561,  3170600, 3403000, '0.472', 'Unemployed',    'Rent',     1],
  [645,  5063000, 2490000, '0.331', 'Self-Employed', 'Rent',     0],
  [798, 10956000, 1535500, '0.142', 'Employed',      'Own',      0],
  [603,  3942500, 4316000, '0.518', 'Unemployed',    'Rent',     1]
];
const ENC_HEAD = ['credit_score', 'annual_income', 'loan_amount', 'debt_to_income_ratio', 'loan_status',
  'employment_status_Self-Employed', 'employment_status_Unemployed', 'home_ownership_Own', 'home_ownership_Rent'];
const ENC_ROWS = [
  [712,  7013500, 1826000, '0.213', 0, false, false, false, false],
  [561,  3170600, 3403000, '0.472', 1, false, true,  false, true ],
  [645,  5063000, 2490000, '0.331', 0, true,  false, false, true ],
  [798, 10956000, 1535500, '0.142', 0, false, false, true,  false],
  [603,  3942500, 4316000, '0.518', 1, false, true,  false, true ]
];

// Five sample trees for the voting visualizer (1 = Default, 0 = Paid)
const TREES = [
  { rule: 'credit_score ≤ 620',                        vote: 1 },
  { rule: 'loan_amount &gt; ₹33,20,000',                   vote: 1 },
  { rule: 'annual_income ≥ ₹24,90,000 and not Unemployed', vote: 0 },
  { rule: 'debt_to_income_ratio &gt; 0.40',            vote: 1 },
  { rule: 'credit_score ≤ 650 and DTI &gt; 0.35',      vote: 1 }
];
const TREE_SVG = '<svg width="26" height="26" viewBox="0 0 26 26" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" aria-hidden="true"><circle cx="13" cy="5" r="2.4"/><path d="M13 7.4v3M13 10.4L7 15M13 10.4L19 15"/><circle cx="7" cy="17.4" r="2.4"/><circle cx="19" cy="17.4" r="2.4"/></svg>';

const IMPORTANCES = [
  ['credit_score', 0.38],
  ['debt_to_income_ratio', 0.27],
  ['employment_status_Unemployed', 0.18],
  ['annual_income', 0.12],
  ['loan_amount', 0.03]
];
const STAGES = ['Load', 'Encode', 'Split', 'Train', 'Evaluate', 'Predict'];

const pct = v => `${v.toFixed(1)}%`;
const MATRIX_TEXT = `[[${CM.tn}  ${CM.fp}]\n [ ${CM.fn}  ${CM.tp}]]`;
const METRICS_TEXT = `Accuracy : ${pct(METRICS.accuracy)}\nPrecision: ${pct(METRICS.precision)}\nRecall   : ${pct(METRICS.recall)}\nF1 Score : ${pct(METRICS.f1)}`;
const PREDICTION_TEXT = `Prediction: 1 (Default Risk)\nDefault Risk Score: ${pct(RISK)}`;

/* ---------------------------------------------------------
   2. Editable Python code for each cell
   --------------------------------------------------------- */
const CELL_CODE = {
  1: `import pandas as pd
df = pd.read_csv("loan_default.csv")
print(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
df.head()`,

  2: `categorical_cols = ["employment_status", "home_ownership"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df_encoded.head()`,

  3: `X = df_encoded.drop("loan_status", axis=1)
y = df_encoded["loan_status"]
print("X =", X.shape)
print("y =", y.shape)`,

  4: `from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Training set: {len(X_train)} samples | Testing set: {len(X_test)} samples")`,

  5: `from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)`,

  6: `# Teaching Simulator — Visualizing Decision Tree Ensemble Voting`,

  7: `import matplotlib.pyplot as plt
import pandas as pd

importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(5).plot(kind='barh')
plt.show()`,

  8: `y_pred = model.predict(X_test)
print("Predicted:", y_pred[:10])
print("Actual:   ", y_test[:10].values)`,

  9: `from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)`,

  10: `from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print(f"Accuracy : {accuracy_score(y_test, y_pred):.1%}")
print(f"Precision: {precision_score(y_test, y_pred):.1%}")
print(f"Recall   : {recall_score(y_test, y_pred):.1%}")
print(f"F1 Score : {f1_score(y_test, y_pred):.1%}")`,

  11: `new_applicant = pd.DataFrame([{
    "credit_score": 580,
    "annual_income": 2905000,
    "loan_amount": 3735000,
    "debt_to_income_ratio": 0.48,
    "employment_status_Self-Employed": 1,
    "employment_status_Unemployed": 0,
    "home_ownership_Own": 0,
    "home_ownership_Rent": 1
}])
prediction = model.predict(new_applicant)
probabilities = model.predict_proba(new_applicant)
label = "Default Risk" if prediction[0] == 1 else "Fully Paid"
print(f"Prediction: {prediction[0]} ({label})")
print(f"Default Risk Score: {probabilities[0][1] * 100:.1f}%")`,

  12: `import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# 1. Load
df = pd.read_csv("loan_default.csv")

# 2. Encode categorical features
df_encoded = pd.get_dummies(df, columns=["employment_status", "home_ownership"], drop_first=True)

# 3. Features and target
X = df_encoded.drop("loan_status", axis=1)
y = df_encoded["loan_status"]

# 4. Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# 5. Train the Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy : {accuracy_score(y_test, y_pred):.1%}")
print(f"Precision: {precision_score(y_test, y_pred):.1%}")
print(f"Recall   : {recall_score(y_test, y_pred):.1%}")
print(f"F1 Score : {f1_score(y_test, y_pred):.1%}")

# 7. Score a new applicant (columns must match the training order)
new_applicant = pd.DataFrame([{
    "credit_score": 580,
    "annual_income": 2905000,
    "loan_amount": 3735000,
    "debt_to_income_ratio": 0.48,
    "employment_status_Self-Employed": 1,
    "employment_status_Unemployed": 0,
    "home_ownership_Own": 0,
    "home_ownership_Rent": 1
}])[X.columns]

prediction = model.predict(new_applicant)
probabilities = model.predict_proba(new_applicant)
label = "Default Risk" if prediction[0] == 1 else "Fully Paid"
print(f"Prediction: {prediction[0]} ({label})")
print(f"Default Risk Score: {probabilities[0][1] * 100:.1f}%")`
};

/* ---------------------------------------------------------
   3. Output templates (HTML strings)
   Each top-level element carries a data-anim attribute that
   tells animateOutput() how to reveal it (see section 5).
   --------------------------------------------------------- */
const tdHTML = v => {
  const cls = v === true ? 't' : v === false ? 'f' : typeof v === 'string' ? 'txt' : '';
  return `<td class="${cls}">${v}</td>`;
};

const tableHTML = (headers, rows, highlight = []) => `
  <div class="tbl-wrap" data-anim="rows">
    <table class="df">
      <thead><tr><th></th>${headers.map((name, i) => `<th class="${highlight.includes(i) ? 'hi' : ''}">${name}</th>`).join('')}</tr></thead>
      <tbody>${rows.map((row, i) => `<tr><td class="idx">${i}</td>${row.map(tdHTML).join('')}</tr>`).join('')}</tbody>
    </table>
  </div>`;

const tileHTML = (kind, value, label) => `<div class="tile ${kind}"><b>${value}</b><small>${label}</small></div>`;

const metricCardHTML = (name, value, desc, key = false) => `
  <div class="metric-card${key ? ' key' : ''}" data-value="${value}">
    <div class="m-name"><span>${name}</span>${key ? '<em>Key for banks</em>' : ''}</div>
    <div class="m-val">0.0%</div>
    <div class="m-bar"><i></i></div>
    <div class="m-desc">${desc}</div>
  </div>`;

const OUTPUTS = {
  1: () => `
    <pre class="term" data-anim="type">Dataset loaded: 1000 rows × 7 columns</pre>
    ${tableHTML(HEAD, ROWS)}`,

  2: () => `
    ${tableHTML(ENC_HEAD, ENC_ROWS, [5, 6, 7, 8])}`,

  3: () => `
    <pre class="term" data-anim="type">X = (1000, 8)\ny = (1000,)</pre>`,

  4: () => `
    <pre class="term" data-anim="type">Training set: 800 samples | Testing set: 200 samples</pre>
    <div data-anim="segs">
      <div class="track">
        <div class="seg neon" data-width="80"></div>
        <div class="seg info" data-width="20"></div>
      </div>
      <div class="legend">
        <span><i style="background:var(--neon)"></i>Train · 800 (80%)</span>
        <span><i style="background:var(--info)"></i>Test · 200 (20%)</span>
      </div>
    </div>`,

  5: () => `
    <div class="mb-block" data-anim="progress">
      <div class="prog-label"><span>Fitting 100 decision trees…</span><span data-count>0 / 100</span></div>
      <div class="track"><div class="seg neon"></div></div>
    </div>
    <pre class="term ok" data-anim="type" data-step="3" data-delay="12">RandomForestClassifier(max_depth=6, n_estimators=100, random_state=42) fitted successfully.</pre>`,

  6: () => `
    <div data-anim="vote">
      <div class="profile">
        <span>credit_score <b>580</b></span><span>annual_income <b>₹29,05,000</b></span><span>loan_amount <b>₹37,35,000</b></span>
        <span>DTI <b>0.48</b></span><span><b>Self-Employed</b></span><span><b>Rent</b></span>
      </div>
      <div class="voters">
        ${TREES.map((t, i) => `
        <button type="button" class="tree" data-vote="${t.vote}" aria-label="Tree ${i + 1} vote. Activate to flip.">
          ${TREE_SVG}
          <div class="tname">Tree ${i + 1}</div>
          <div class="rule">${t.rule}</div>
          <span class="vote"></span>
        </button>`).join('')}
      </div>
      <div class="tally-wrap">
        <div class="track"><div class="seg bad"></div><div class="seg neon"></div></div>
        <div class="legend"></div>
      </div>
      <div class="verdict"></div>
      <p class="note">Illustrative applicant. Only 5 of the 100 trees are shown. Click a tree to flip its vote and watch the majority change.</p>
    </div>`,

  7: () => `
    <div class="fi" data-anim="bars">
      <div class="fi-title">Top 5 risk factors (feature importance)</div>
      ${IMPORTANCES.map(([name, value]) => `
      <div class="fi-row">
        <div class="fi-name">${name}</div>
        <div class="fi-track"><div class="fi-fill" data-width="${(value / IMPORTANCES[0][1] * 100).toFixed(1)}"></div></div>
        <div class="fi-val">${value.toFixed(2)}</div>
      </div>`).join('')}
      <div class="fi-axis">Importance (share of total impurity reduction)</div>
    </div>`,

  8: () => `
    <pre class="term" data-anim="type">Predicted: [0 1 0 0 1 0 0 1 0 0]\nActual:    [0 1 0 0 1 0 1 1 0 0]</pre>
    <p class="note" data-anim="fade"><b>9 of 10 correct.</b> Position 6 was a missed default: predicted Paid, actually Default.</p>`,

  9: () => `
    <pre class="term" data-anim="type">${MATRIX_TEXT}</pre>
    <div class="cm" data-anim="tiles">
      <div></div><div class="axis">Predicted Paid</div><div class="axis">Predicted Default</div>
      <div class="rowlab">Actual Paid</div>
      ${tileHTML('good', CM.tn, 'True Negative (TN)')}${tileHTML('bad', CM.fp, 'False Positive (FP)')}
      <div class="rowlab">Actual Default</div>
      ${tileHTML('bad', CM.fn, 'False Negative (FN)')}${tileHTML('good', CM.tp, 'True Positive (TP)')}
    </div>`,

  10: () => `
    <pre class="term" data-anim="type">${METRICS_TEXT}</pre>
    <div class="metric-grid" data-anim="metrics">
      ${metricCardHTML('Accuracy', METRICS.accuracy, 'Overall correctness')}
      ${metricCardHTML('Precision', METRICS.precision, 'Reliability when flagging risk')}
      ${metricCardHTML('Recall', METRICS.recall, 'Percentage of actual defaults caught', true)}
      ${metricCardHTML('F1 Score', METRICS.f1, 'Balance between precision and recall')}
    </div>`,

  11: () => `
    <pre class="term" data-anim="type">${PREDICTION_TEXT}</pre>
    <div class="gauge" data-anim="gauge" data-target="${RISK}">
      <div class="gauge-top"><span class="gauge-num">0.0%</span><span class="gauge-tag">High default risk</span></div>
      <div class="gauge-bar"><div class="gauge-mark"></div></div>
      <div class="gauge-scale"><span>0%</span><span>50%</span><span>100%</span></div>
    </div>`,

  12: () => `
    <div class="pipe" data-anim="chips">
      ${STAGES.map((s, i) => `${i ? '<span class="arrow">→</span>' : ''}<span class="chip">${s}</span>`).join('')}
    </div>
    <pre class="term" data-anim="type" data-step="3" data-delay="12">${MATRIX_TEXT}\n${METRICS_TEXT}\n${PREDICTION_TEXT}</pre>
    <pre class="term ok" data-anim="type">✔ Pipeline executed successfully: load, encode, split, train, evaluate, predict.</pre>`
};

/* ---------------------------------------------------------
   4. Explanation cards: "⚙ What's happening in the backend?"
   Appended under every output by runCell().
   --------------------------------------------------------- */
const point = (tag, text, kind = '') =>
  `<li class="ex-point ex-item"><span class="ex-tag ${kind}">${tag}</span><span>${text}</span></li>`;

const explainCard = (intro, points) => `
  <aside class="explain" data-anim="explain" aria-label="Backend explanation">
    <div class="explain-head"><span aria-hidden="true">⚙</span> What's happening in the backend?</div>
    <p class="ex-intro ex-item">${intro}</p>
    <ul class="ex-points">${points.join('')}</ul>
  </aside>`;

const EXPLANATIONS = {
  1: () => explainCard(
    `We load raw loan applicant data into a <b>Pandas DataFrame</b>. Each row represents a historical borrower. <code>loan_status</code> is our target label (1 = Default, 0 = Fully Paid). Real-world data needs clean structuring before machine learning algorithms can interpret it.`,
    [
      point('Logic', `<code>pd.read_csv</code> parses the file into a table with 1,000 rows and 7 columns. <code>df.head()</code> previews the first 5 rows.`),
      point('Data types', `4 numeric features (<code>credit_score</code>, <code>annual_income</code>, <code>loan_amount</code>, <code>debt_to_income_ratio</code>), 2 text categories, and 1 target label.`),
      point('Business impact', `The model can only learn from the history it is shown, so the quality and fairness of these records set the ceiling for every later step.`)
    ]),

  2: () => explainCard(
    `Machine learning models only process numbers. <b>One-Hot Encoding</b> converts categorical text (like 'Employed' vs 'Unemployed') into 0s and 1s. <code>drop_first=True</code> removes one dummy column per feature to prevent multi-collinearity (redundant data).`,
    [
      point('Logic', `Each text value becomes its own True/False column. <code>employment_status</code> (3 values) turns into 2 columns and <code>home_ownership</code> (3 values) into 2. <b>Employed</b> and <b>Mortgage</b> become the baseline: all dummies False means that category.`),
      point('Why not 1, 2, 3?', `Coding Rent = 1, Own = 2, Mortgage = 3 would tell the model that Own is "twice" Rent. One-hot columns avoid inventing an order that does not exist.`),
      point('Good to know', `Random Forests tolerate redundant columns anyway, but dropping the extra one is standard practice and keeps the same data usable with linear models.`),
      point('Result', `4 numeric + 4 dummy + the target = 9 columns. The highlighted green columns are the new ones.`)
    ]),

  3: () => explainCard(
    `We separate the dataset into inputs (<code>X</code>) containing applicant metrics, and target labels (<code>y</code>) containing default history. The model uses <code>X</code> to learn patterns that predict <code>y</code>.`,
    [
      point('Logic', `<code>drop("loan_status", axis=1)</code> removes that column (<code>axis=1</code> means columns, not rows). What is left is 4 numeric + 4 dummy features = 8 columns.`),
      point('Reading the shapes', `<code>(1000, 8)</code> is 1,000 applicants × 8 features. <code>(1000,)</code> is one label per applicant.`),
      point('Business impact', `The outcome is kept out of <code>X</code> on purpose. If the model could see it, that would be <b>data leakage</b>: perfect scores on paper, useless on real applicants.`)
    ]),

  4: () => explainCard(
    `To test if our model actually works on future applicants, we hold back 20% of data (<code>X_test</code>). <b>Stratification</b> ensures both training and testing sets maintain the same proportion of default vs non-default loans.`,
    [
      point('Math', `1,000 × 0.20 = 200 test rows, leaving 800 for training. <code>random_state=42</code> fixes the shuffle so everyone gets the identical split.`),
      point('Why stratify?', `Defaults are the minority class. <code>stratify=y</code> keeps the same default share in both sets, so neither one is accidentally easier than the other.`),
      point('Business impact', `The 200 held-back rows play the role of future applicants. Scoring them gives an honest estimate of performance before real money is lent.`)
    ]),

  5: () => explainCard(
    `A Random Forest creates <b>100 individual decision trees</b> (<code>n_estimators=100</code>). Each tree is trained on a random sample of rows and features (<b>bagging</b>). Restricting <code>max_depth=6</code> prevents trees from becoming overly complex and overfitting.`,
    [
      point('Bagging', `Each tree trains on a bootstrap sample: 800 rows drawn with replacement, so about 63% of the distinct rows appear in any one tree. At every split a tree also considers only a random subset of the features (about 2 of 8 by default).`),
      point('Splitting rule', `Splits are chosen to minimise Gini impurity, <code>1 − p(default)² − p(paid)²</code>. A pure group scores 0 and a 50/50 group scores 0.5.`),
      point('Depth limit', `<code>max_depth=6</code> allows at most 2⁶ = 64 leaves per tree: enough for useful rules, too few to memorise individual borrowers.`),
      point('Business impact', `A model that memorises past borrowers looks perfect in training and fails on new applicants. Limiting depth keeps the rules general.`)
    ]),

  6: () => explainCard(
    `Unlike a single Decision Tree, Random Forest uses <b>ensemble voting</b>. Each tree casts an independent vote on whether a borrower will default. The majority vote wins, drastically reducing prediction variance and errors.`,
    [
      point('Math', `In the picture, 4 of 5 trees say Default: 4 ÷ 5 = 80%. The real forest has 100 trees, and scikit-learn averages each tree's predicted probability (a "soft" vote) rather than counting labels, but the idea is the same.`),
      point('Why it works', `Each tree sees different rows and features, so their mistakes differ and partly cancel out. Many noisy but independent opinions are more stable than any single one.`),
      point('Try it', `Click any tree above to flip its vote. With 5 voters, 3 or more Default votes decide the verdict.`),
      point('Business impact', `One quirky tree cannot decline a good customer on its own. It has to persuade the majority.`)
    ]),

  7: () => explainCard(
    `Random Forest measures feature importance by calculating how much each feature reduces impurity (<b>Gini value</b>) across all 100 trees. Credit score and debt-to-income ratio are identified as the strongest drivers of loan default.`,
    [
      point('Math', `Every split lowers impurity a little. A feature's importance is the total impurity drop from all splits that used it (weighted by how many applicants reached each split), averaged over the 100 trees and scaled so all 8 features sum to 1.0.`),
      point('Reading the chart', `The five bars add up to 0.98. The last 0.02 is shared by the three features not shown. <code>credit_score</code> and <code>debt_to_income_ratio</code> alone carry 0.65 of the total.`),
      point('Caution', `Importance shows what the model relied on, not what causes default. Use it as a starting point for risk review, not as proof.`),
      point('Business impact', `Credit teams can see which applicant attributes drive decisions and explain them to reviewers and customers.`)
    ]),

  8: () => explainCard(
    `We pass the unseen test applicants (<code>X_test</code>) to the trained model. Comparing predicted decisions against actual historical outcomes lets us evaluate <b>true predictive performance</b>.`,
    [
      point('Logic', `For every test row, all 100 trees vote and the majority label is returned. <code>y_pred[:10]</code> shows the first 10 predictions. <code>.values</code> turns the pandas Series into a plain array for printing.`),
      point('Reading the result', `9 of 10 match. Position 6 was predicted Paid (0) but actually defaulted (1): a missed default.`),
      point('Business impact', `Ten rows are only a glance. The next cells score all 200 held-out applicants.`)
    ]),

  9: () => explainCard(
    `The confusion matrix tracks exact prediction errors:`,
    [
      point(`True Positives (${CM.tp})`, `Correctly flagged high-risk default borrowers.`, 'good'),
      point(`False Positives (${CM.fp})`, `Approved borrowers flagged as default risk (false alarm).`, 'bad'),
      point(`False Negatives (${CM.fn})`, `Defaulted borrowers missed by the model (a high risk for the bank).`, 'bad'),
      point(`True Negatives (${CM.tn})`, `Correctly approved safe borrowers.`, 'good'),
      point('How to read it', `Rows are the actual outcome and columns are the prediction. scikit-learn lists <code>[[TN, FP], [FN, TP]]</code>, with Default (1) as the "positive" class. ${CM.tn} + ${CM.fp} + ${CM.fn} + ${CM.tp} = 200, one cell for every test applicant.`),
      point('Business impact', `A false negative is money lent that is never repaid. A false positive is a good customer turned away and interest income lost. The bank has to weigh those two costs.`)
    ]),

  10: () => explainCard(
    `One metric is never enough in fintech. Accuracy shows general performance, but <b>Recall</b> is critical because a False Negative (missing a loan default) costs the bank significantly more than a false alarm.`,
    [
      point('Formulas', `Accuracy = (TP + TN) ÷ all applicants. Precision = TP ÷ (TP + FP). Recall = TP ÷ (TP + FN). F1 = 2 × Precision × Recall ÷ (Precision + Recall).`),
      point('Worked example', `Accuracy from the matrix: (${CM.tp} + ${CM.tn}) ÷ 200 = ${pct(METRICS.accuracy)}.`),
      point('Business impact', `Lowering the decision threshold below 50% catches more defaults (higher recall) at the cost of more false alarms (lower precision). The bank chooses the balance.`)
    ]),

  11: () => explainCard(
    `For real-world deployment, the model returns a <b>probability percentage</b> (84.0%) alongside the final decision label. Banks use this score to automatically decline high-risk applicants or flag them for manual underwriting.`,
    [
      point('Logic', `<code>predict_proba</code> returns two numbers per applicant: P(Paid) and P(Default). Each of the 100 trees contributes the default share of the leaf this applicant lands in, and the forest averages them to 0.84. <code>predict</code> picks Default because 0.84 is above 0.50.`),
      point('The applicant', `Credit score 580, income ₹29,05,000, loan ₹37,35,000, debt-to-income ratio 0.48, Self-Employed, renting. A low score, a high DTI and a loan larger than annual income all point the same way.`),
      point('Business impact', `A score can route applicants: approve the lowest-risk automatically, send the middle band to manual underwriting, and decline the highest-risk cases.`)
    ]),

  12: () => explainCard(
    `This complete pipeline demonstrates <b>end-to-end Machine Learning execution</b>: from reading raw data to encoding, splitting, model fitting, evaluation, and live inference.`,
    [
      point('The flow', `Load → Encode → Split → Train → Evaluate → Predict. Each stage feeds the next, and the whole thing runs top to bottom in one script.`),
      point('Consistency', `New applicants must be encoded the same way and have their columns in the same order as the training data (<code>[X.columns]</code>). Otherwise the model silently reads the wrong column as the wrong feature.`),
      point('Business impact', `In production this becomes a service: applicant data goes in, a risk score and decision come out. Retraining on fresh data keeps it accurate as borrower behaviour and the economy change.`)
    ])
};

/* ---------------------------------------------------------
   5. Output animators — one per data-anim value.
      Each receives (ctx, node) and resolves when finished.
   --------------------------------------------------------- */
const ANIMATORS = {
  // Terminal text with a typewriter effect
  async type(ctx, node) {
    const text = node.textContent;
    const step = REDUCE_MOTION ? text.length : Number(node.dataset.step || 2);
    const delay = Number(node.dataset.delay || 16);
    const textNode = document.createTextNode('');
    const caret = document.createElement('span');
    caret.className = 'caret';
    node.textContent = '';
    node.append(textNode, caret);
    for (let i = 0; i < text.length; i += step) {
      textNode.data = text.slice(0, i + step);
      await ctx.sleep(delay);
    }
    textNode.data = text;
    caret.remove();
  },

  // Plain fade-in (CSS animation on .note)
  async fade(ctx) {
    await ctx.sleep(250);
  },

  // Table rows fade in one by one
  async rows(ctx, node) {
    for (const tr of node.querySelectorAll('tbody tr')) {
      await ctx.sleep(130);
      tr.classList.add('show');
    }
  },

  // Horizontal segments growing from 0% (train/test split)
  async segs(ctx, node) {
    await nextFrame();
    for (const seg of node.querySelectorAll('.seg')) {
      seg.style.width = seg.dataset.width + '%';
      await ctx.sleep(500);
    }
  },

  // Model-fitting progress bar
  async progress(ctx, node) {
    const seg = node.querySelector('.seg');
    const count = node.querySelector('[data-count]');
    seg.style.transition = 'width .12s linear';
    for (let i = 0; i <= 100; i += 5) {
      seg.style.width = i + '%';
      count.textContent = i + ' / 100';
      await ctx.sleep(55);
    }
  },

  // Feature-importance bars growing from 0% to their target length
  async bars(ctx, node) {
    await nextFrame();
    for (const fill of node.querySelectorAll('.fi-fill')) {
      await ctx.sleep(160);
      fill.style.width = fill.dataset.width + '%';
    }
  },

  // Confusion-matrix tiles popping in
  async tiles(ctx, node) {
    for (const tile of node.querySelectorAll('.tile')) {
      await ctx.sleep(260);
      tile.classList.add('on');
    }
  },

  // Scorecard cards: fade in, bar fills, number counts up
  async metrics(ctx, node) {
    for (const card of node.querySelectorAll('.metric-card')) {
      await ctx.sleep(200);
      card.classList.add('on');
      const target = Number(card.dataset.value);
      const val = card.querySelector('.m-val');
      card.querySelector('.m-bar i').style.width = target + '%';
      for (let v = 0; v < target; v += 5) {
        val.textContent = v.toFixed(1) + '%';
        await ctx.sleep(24);
      }
      val.textContent = target.toFixed(1) + '%';
    }
  },

  // Default-risk gauge with count-up
  async gauge(ctx, node) {
    const target = Number(node.dataset.target);
    const num = node.querySelector('.gauge-num');
    await nextFrame();
    node.querySelector('.gauge-mark').style.left = target + '%';
    for (let v = 0; v < target; v += 4) {
      num.textContent = v.toFixed(1) + '%';
      await ctx.sleep(30);
    }
    num.textContent = target.toFixed(1) + '%';
  },

  // Pipeline stage chips lighting up in order
  async chips(ctx, node) {
    for (const chip of node.querySelectorAll('.chip')) {
      await ctx.sleep(380);
      chip.classList.add('on');
    }
  },

  // Backend explanation card: intro and each point fade in one after another
  async explain(ctx, node) {
    for (const item of node.querySelectorAll('.ex-item')) {
      await ctx.sleep(item.classList.contains('ex-intro') ? 150 : 260);
      item.classList.add('show');
    }
  },

  // Random Forest majority vote: trees reveal their votes, then become clickable
  async vote(ctx, node) {
    const cards = [...node.querySelectorAll('.tree')];
    const votes = cards.map(c => Number(c.dataset.vote));
    const segD = node.querySelector('.tally-wrap .seg.bad');
    const segP = node.querySelector('.tally-wrap .seg.neon');
    const legend = node.querySelector('.tally-wrap .legend');
    const verdict = node.querySelector('.verdict');
    const total = votes.length;

    const paintVote = i => {
      const badge = cards[i].querySelector('.vote');
      badge.className = 'vote show ' + (votes[i] ? 'def' : 'paid');
      badge.textContent = votes[i] ? 'Default' : 'Paid';
    };
    const paintTally = () => {
      const d = votes.reduce((a, b) => a + b, 0);
      const p = total - d;
      segD.style.width = (d / total * 100) + '%';
      segP.style.width = (p / total * 100) + '%';
      legend.innerHTML =
        `<span><i style="background:var(--danger)"></i>Default · ${d}</span>` +
        `<span><i style="background:var(--neon)"></i>Paid · ${p}</span>`;
      const defaultWins = d > p;
      verdict.className = 'verdict ' + (defaultWins ? 'def' : 'paid');
      verdict.textContent =
        `Majority Vote: ${defaultWins ? 'Default Risk' : 'Fully Paid'} (${Math.round(Math.max(d, p) / total * 100)}% confidence)`;
    };

    verdict.style.visibility = 'hidden';
    await nextFrame();

    let defaults = 0;
    for (let i = 0; i < total; i++) {
      await ctx.sleep(520);
      paintVote(i);
      defaults += votes[i];
      segD.style.width = (defaults / total * 100) + '%';
      segP.style.width = ((i + 1 - defaults) / total * 100) + '%';
    }
    await ctx.sleep(500);
    paintTally();
    verdict.style.visibility = 'visible';

    cards.forEach((card, i) => {
      card.classList.add('live');
      card.addEventListener('click', () => {
        votes[i] = votes[i] ? 0 : 1;
        paintVote(i);
        paintTally();
      });
    });
  }
};

// Reveals the top-level blocks of an output one after another
async function animateOutput(ctx) {
  const blocks = [...ctx.body.children];
  blocks.forEach(b => b.classList.add('pending'));
  for (const block of blocks) {
    ctx.guard();
    block.classList.remove('pending');
    const animate = ANIMATORS[block.dataset.anim] || ANIMATORS.fade;
    await animate(ctx, block);
  }
}

/* ---------------------------------------------------------
   6. runCell(cellId) + notebook controls
   --------------------------------------------------------- */
const cellEls = {};   // cellId -> { section, textarea, prompt, out, body, token }
let busyCount = 0;

function setBusy(delta) {
  busyCount = Math.max(0, busyCount + delta);
  const kernel = document.getElementById('kernel');
  const label = document.getElementById('kernel-text');
  if (!kernel || !label) return;
  kernel.classList.toggle('busy', busyCount > 0);
  label.textContent = busyCount > 0 ? 'Python 3 · kernel busy' : 'Python 3 · kernel idle';
}

async function runCell(cellId) {
  const el = cellEls[cellId];
  const template = OUTPUTS[cellId];
  if (!el || !template) return;

  const token = ++el.token;
  const ctx = {
    body: el.body,
    guard: () => { if (el.token !== token) throw CANCEL; },
    sleep: async ms => { await wait(ms); if (el.token !== token) throw CANCEL; }
  };

  setBusy(1);
  el.section.classList.add('running');
  el.prompt.textContent = 'In [*]:';
  el.out.hidden = false;
  el.body.innerHTML = '';

  let cancelled = false;
  try {
    await ctx.sleep(380);                                   // short "executing…" pause
    const explanation = EXPLANATIONS[cellId];
    el.body.innerHTML = template() + (explanation ? explanation() : '');   // 1. inject output + explanation card
    await animateOutput(ctx);                                               // 2. reveal block by block
  } catch (err) {
    if (err === CANCEL) cancelled = true; else console.error(err);
  }

  setBusy(-1);
  if (cancelled) return;                                    // a newer run or a clear took over
  el.section.classList.remove('running');
  el.prompt.textContent = `In [${cellId}]:`;
}

const cellIds = () => Object.keys(cellEls).map(Number).sort((a, b) => a - b);

async function runAll() {
  const btn = document.getElementById('run-all');
  if (btn) btn.disabled = true;
  for (const id of cellIds()) {
    cellEls[id].section.scrollIntoView({ behavior: REDUCE_MOTION ? 'auto' : 'smooth', block: 'start' });
    await runCell(id);
    await wait(260);
  }
  if (btn) btn.disabled = false;
}

function clearCell(cellId) {
  const el = cellEls[cellId];
  el.token++;
  el.body.innerHTML = '';
  el.out.hidden = true;
  el.section.classList.remove('running');
  el.prompt.textContent = `In [${cellId}]:`;
}

function clearAll() {
  cellIds().forEach(clearCell);
}

async function copyCode(cellId, btn) {
  const text = cellEls[cellId].textarea.value;
  try {
    await navigator.clipboard.writeText(text);
  } catch (err) {
    const ta = document.createElement('textarea');
    ta.value = text;
    ta.style.cssText = 'position:fixed;opacity:0';
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (_) { /* ignore */ }
    ta.remove();
  }
  const original = btn.textContent;
  btn.classList.add('copied');
  btn.textContent = 'Copied';
  setTimeout(() => { btn.classList.remove('copied'); btn.textContent = original; }, 1400);
}

/* ---------------------------------------------------------
   7. Editor setup & boot
   --------------------------------------------------------- */
// Python syntax highlighting for the code editor
const TOKEN = /(#[^\n]*)|([fF]?"(?:\\.|[^"\\\n])*"|[fF]?'(?:\\.|[^'\\\n])*')|\b(import|from|as|def|return|for|in|if|else|and|or|not)\b|\b(True|False|None)\b|\b(\d+(?:\.\d+)?)\b|\b([A-Za-z_]\w*)(?=\()/g;

function highlight(source) {
  const escaped = source.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return escaped.replace(TOKEN, (match, comment, str, keyword, constant, number) => {
    const kind = comment ? 'c' : str ? 's' : keyword ? 'k' : constant ? 'b' : number ? 'n' : 'f';
    return `<span class="tok-${kind}">${match}</span>`;
  }) + '\n';
}

function initNotebook() {
  document.querySelectorAll('.cell[data-cell]').forEach(section => {
    const id = Number(section.dataset.cell);
    const textarea = section.querySelector('textarea');
    const codeEl = section.querySelector('.editor code');
    const sync = () => { codeEl.innerHTML = highlight(textarea.value); };

    textarea.value = CELL_CODE[id] || '';
    sync();

    textarea.addEventListener('input', sync);
    textarea.addEventListener('blur', () => { delete textarea.dataset.free; });
    textarea.addEventListener('keydown', e => {
      if (e.key === 'Escape') {
        textarea.dataset.free = '1';                         // Esc, then Tab moves focus on
      } else if (e.key === 'Tab' && !e.shiftKey && !textarea.dataset.free) {
        e.preventDefault();
        textarea.setRangeText('    ', textarea.selectionStart, textarea.selectionEnd, 'end');
        sync();
      } else if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        runCell(id);
      }
    });

    cellEls[id] = {
      section,
      textarea,
      prompt: section.querySelector('.prompt'),
      out: section.querySelector('.out'),
      body: section.querySelector('.out-body'),
      token: 0
    };
  });
}

initNotebook();

/* ---------------------------------------------------------
   8. Live Risk Calculator & Playground
   Same logistic-regression-style model used to generate the
   training data, evaluated client-side for instant feedback.
   --------------------------------------------------------- */
const CALC_DEFAULTS = { credit: 680, income: 5395000, loan: 1245000, dti: 30, employment: 'Employed', home: 'Mortgage' };

// Indian Rupee formatting with lakh/crore digit grouping (e.g. ₹53,95,000)
const money = n => '₹' + Math.round(n).toLocaleString('en-IN');

// Mirrors the Python data-generation formula in generate_loan_default.py
function computeRisk({ credit, income, loan, dti, employment, home }) {
  const creditZ = (credit - 680) / 80;
  const dtiRatio = dti / 100;
  const dtiZ = (dtiRatio - 0.25) / 0.12;
  const ltiRatio = loan / income;
  const ltiZ = (ltiRatio - 0.35) / 0.25;

  const terms = [
    { key: 'credit', value: -1.0 * creditZ, label: creditZ >= 0 ? `High Credit Score (${credit})` : `Low Credit Score (${credit})` },
    { key: 'dti', value: 0.9 * dtiZ, label: dtiZ >= 0 ? `High Debt-to-Income (${dti}%)` : `Low Debt-to-Income (${dti}%)` },
    { key: 'lti', value: 0.3 * ltiZ, label: ltiRatio >= 0.35 ? 'Large Loan vs. Income' : 'Small Loan vs. Income' },
    { key: 'unemployed', value: employment === 'Unemployed' ? 1.5 : 0, label: 'Unemployed' },
    { key: 'self', value: employment === 'Self-Employed' ? 0.25 : 0, label: 'Self-Employed' },
    { key: 'rent', value: home === 'Rent' ? 0.25 : 0, label: 'Renting (no home equity)' },
    { key: 'own', value: home === 'Own' ? -0.2 : 0, label: 'Owns Home Outright' }
  ];

  const logOdds = -2.2 + terms.reduce((sum, t) => sum + t.value, 0);
  const prob = 1 / (1 + Math.exp(-logOdds));

  // First-order approximation of each term's contribution in probability points,
  // via the logistic derivative p(1-p) — good for an at-a-glance breakdown.
  const slope = prob * (1 - prob) * 100;
  const contributions = terms
    .filter(t => Math.abs(t.value) > 0.005)
    .map(t => ({ label: t.label, pct: t.value * slope }))
    .sort((a, b) => Math.abs(b.pct) - Math.abs(a.pct))
    .slice(0, 5);

  return { prob: Math.max(0, Math.min(1, prob)), contributions };
}

function zoneFor(pct) {
  if (pct <= 30) return { cls: 'low', color: 'var(--mint)', text: 'Low Risk — Auto Approved' };
  if (pct <= 60) return { cls: 'medium', color: '#fbbf24', text: 'Moderate Risk — Manual Review' };
  return { cls: 'high', color: 'var(--crimson)', text: 'High Risk — Declined' };
}

function renderGauge(pct) {
  const zone = zoneFor(pct);
  const dialFill = document.getElementById('dial-fill');
  const needle = document.getElementById('dial-needle');
  const num = document.getElementById('dial-num');
  const badge = document.getElementById('calc-verdict');
  const badgeText = document.getElementById('calc-verdict-text');

  const ARC_LENGTH = 298; // ≈ π·95, the traced semicircle's length
  dialFill.style.strokeDashoffset = String(ARC_LENGTH - (pct / 100) * ARC_LENGTH);
  dialFill.style.stroke = zone.color;
  needle.style.transform = `rotate(${-90 + (pct / 100) * 180}deg)`;
  num.textContent = Math.round(pct) + '%';
  num.style.color = zone.color;

  badge.className = 'verdict-badge ' + zone.cls;
  badgeText.textContent = zone.text;
}

function renderContributions(contributions) {
  const root = document.getElementById('contrib-list');
  if (!contributions.length) {
    root.innerHTML = '<p class="note">No single factor stands out — this applicant is close to the baseline profile.</p>';
    return;
  }
  const maxAbs = Math.max(...contributions.map(c => Math.abs(c.pct)), 1);
  root.innerHTML = contributions.map(c => {
    const up = c.pct >= 0;
    const dir = up ? 'up' : 'down';
    const sign = up ? '+' : '−';
    const width = Math.min(100, Math.abs(c.pct) / maxAbs * 100).toFixed(0);
    return `
      <div class="contrib-row">
        <span class="contrib-label">${c.label}</span>
        <span class="contrib-value ${dir}">${sign}${Math.abs(c.pct).toFixed(0)}% Risk</span>
        <div class="contrib-track"><div class="contrib-fill ${dir}" style="width:${width}%"></div></div>
      </div>`;
  }).join('');
}

function readCalculatorInputs() {
  return {
    credit: Number(document.getElementById('in-credit').value),
    income: Number(document.getElementById('in-income').value),
    loan: Number(document.getElementById('in-loan').value),
    dti: Number(document.getElementById('in-dti').value),
    employment: document.getElementById('in-employment').value,
    home: document.getElementById('in-home').value
  };
}

function updateCalculator() {
  const inputs = readCalculatorInputs();

  document.getElementById('v-credit').textContent = inputs.credit;
  document.getElementById('v-income').textContent = money(inputs.income);
  document.getElementById('v-loan').textContent = money(inputs.loan);
  document.getElementById('v-dti').textContent = inputs.dti + '%';

  const { prob, contributions } = computeRisk(inputs);
  renderGauge(prob * 100);
  renderContributions(contributions);
}

function resetCalculator() {
  document.getElementById('in-credit').value = CALC_DEFAULTS.credit;
  document.getElementById('in-income').value = CALC_DEFAULTS.income;
  document.getElementById('in-loan').value = CALC_DEFAULTS.loan;
  document.getElementById('in-dti').value = CALC_DEFAULTS.dti;
  document.getElementById('in-employment').value = CALC_DEFAULTS.employment;
  document.getElementById('in-home').value = CALC_DEFAULTS.home;
  updateCalculator();
}

function initCalculator() {
  const root = document.getElementById('calculator');
  if (!root) return;
  root.querySelectorAll('input, select').forEach(el => {
    el.addEventListener('input', updateCalculator);
    el.addEventListener('change', updateCalculator);
  });
  updateCalculator();
}

initCalculator();
