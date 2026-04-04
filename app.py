import streamlit as st
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import re
import subprocess
import shutil
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score
from collections import defaultdict
from sklearn.preprocessing import label_binarize, StandardScaler
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(page_title="Gearbox Fault Diagnosis", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# CONFIGURATION DICTIONARY
# ============================================================
# This dictionary holds all the specific settings for both modes
APP_CONFIG = {
    "Epicyclic Gearbox": {
        "repo_url": "https://github.com/NoDhruvHere/Final-Year-project-Vibration-DATA.git",
        "repo_name": "Final-Year-project-Vibration-DATA",
        "data_path": "Final-Year-project-Vibration-DATA/CVS",
        "folders": [
            {'path': 'Healthy', 'label': 0, 'name': 'Healthy'},
            {'path': os.path.join('Stage 1 excel', 'Planet surface crack'), 'label': 1, 'name': 'Planet surface crack (S1)'},
            {'path': os.path.join('Stage 1 excel', 'Ring Broken tooth'), 'label': 2, 'name': 'Ring Broken tooth (S1)'},
            {'path': os.path.join('Stage 1 excel', 'Sun Chipped tooth'), 'label': 3, 'name': 'Sun Chipped tooth (S1)'},
            {'path': os.path.join('Stage 2', 'Planet gear defect 75% Chipped tooth in taper'), 'label': 4, 'name': 'Planet gear defect 75% (S2)'},
            {'path': os.path.join('Stage 2', 'Ring gear one tooth Missing'), 'label': 5, 'name': 'Ring gear one tooth Missing (S2)'},
            {'path': os.path.join('Stage 2', 'Sun Gear Defect Chipped Tooth 2nd'), 'label': 6, 'name': 'Sun Gear Defect Chipped Tooth 2nd (S2)'},
            {'path': os.path.join('Stage 3', 'Planet 2 Broken tooth 180'), 'label': 7, 'name': 'Planet 2 Broken tooth 180 (S3)'},
            {'path': os.path.join('Stage 3', 'RING 2 TOOTH 120'), 'label': 8, 'name': 'RING 2 TOOTH 120 (S3)'},
            {'path': os.path.join('Stage 3', 'Sun Gear 2 tooth taper crack'), 'label': 9, 'name': 'Sun Gear 2 tooth taper crack (S3)'}
        ],
        "class_names": [
            'Healthy', 'Planet crack (S1)', 'Ring Broken (S1)', 'Sun Chipped (S1)',
            'Planet 75% (S2)', 'Ring Missing (S2)', 'Sun Chipped 2nd (S2)',
            'Planet 2 Broken (S3)', 'Ring 2 Tooth (S3)', 'Sun 2 Taper (S3)'
        ]
    },
    "2-Stage Gearbox (Spur)": {
        "repo_url": "https://github.com/NoDhruvHere/Vibration-Project-Data.git",
        "repo_name": "Vibration-Project-Data",
        "data_path": "Vibration-Project-Data/CSV",
        "folders": [
            # Labels shifted to 0-based here for consistency
            {'path': os.path.join('HEALTHY GEAR', 'DE side'), 'label': 0, 'name': 'Healthy'},
            {'path': os.path.join('HEALTHY GEAR', 'Stage 1'), 'label': 0, 'name': 'Healthy'},
            {'path': os.path.join('HEALTHY GEAR', 'Stage 2'), 'label': 0, 'name': 'Healthy'},
            {'path': os.path.join('ADDENDUM WEAR', 'DE side'), 'label': 1, 'name': 'ADDENDUM WEAR'},
            {'path': os.path.join('ADDENDUM WEAR', 'Stage 1'), 'label': 1, 'name': 'ADDENDUM WEAR'},
            {'path': os.path.join('ADDENDUM WEAR', 'Stage 2'), 'label': 1, 'name': 'ADDENDUM WEAR'},
            {'path': os.path.join('BROKEN TOOTH', 'DE side'), 'label': 2, 'name': 'BROKEN TOOTH'},
            {'path': os.path.join('BROKEN TOOTH', 'Stage 1'), 'label': 2, 'name': 'BROKEN TOOTH'},
            {'path': os.path.join('BROKEN TOOTH', 'Stage 2'), 'label': 2, 'name': 'BROKEN TOOTH'},
            {'path': os.path.join('PITTING', 'DE side'), 'label': 3, 'name': 'PITTING'},
            {'path': os.path.join('PITTING', 'Stage 1'), 'label': 3, 'name': 'PITTING'},
            {'path': os.path.join('PITTING', 'Stage 2'), 'label': 3, 'name': 'PITTING'}
        ],
        "class_names": ['Healthy', 'ADDENDUM WEAR', 'BROKEN TOOTH', 'PITTING']
    }
}

# Global Settings
WINDOW_SIZE = 1000
STRIDE = 400
AUGMENTATION_FACTOR = 3

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def augment_signal(signal, aug_type):
    if aug_type == 'noise':
        noise = np.random.normal(0, 0.01 * np.std(signal), len(signal))
        return signal + noise
    elif aug_type == 'scale':
        factor = np.random.uniform(0.9, 1.1)
        return signal * factor
    elif aug_type == 'jitter':
        jitter = np.random.uniform(-0.05, 0.05, len(signal))
        return signal + jitter * np.std(signal)
    elif aug_type == 'time_shift':
        shift = np.random.randint(-50, 50)
        return np.roll(signal, shift)
    return signal

def extract_features(signal):
    features = []
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    rms_val = np.sqrt(np.mean(signal**2))

    # Robust Skew/Kurtosis handling
    if std_val < 1e-10:
        skew_val = 0.0
        kurt_val = 0.0
    else:
        skew_val = stats.skew(signal)
        kurt_val = stats.kurtosis(signal)

    features.extend([mean_val, std_val, max_val, min_val, rms_val, skew_val, kurt_val,
                     np.mean(np.abs(signal)), np.sum(signal**2)])

    fft_vals = np.abs(np.fft.fft(signal))
    freqs = np.fft.fftfreq(len(signal))
    
    if len(fft_vals) > 2:
        dominant_idx = np.argmax(fft_vals[1:len(fft_vals)//2]) + 1
    else:
        dominant_idx = 0
        
    features.append(freqs[dominant_idx])
    features.append(np.sum(fft_vals))
    features.append(np.std(fft_vals))
    return np.array(features)

def extract_speed_from_file(file_path):
    match = re.search(r'(\d+)\s*(?:Hz|RPM|hz|rpm)\b', file_path)
    if match: return int(match.group(1))
    match = re.search(r'[-_](\d+)\.csv$', file_path, re.IGNORECASE)
    if match: return int(match.group(1))
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if "Unnamed" in line: break
                match = re.search(r'(?:speed|freq|frequency|hertz|hz|h\.z\.?)\s*[:=]\s*(\d+)', line, re.IGNORECASE)
                if match: return int(match.group(1))
    except: pass
    return 0

def calculate_specificity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(len(cm)):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - tp)
        if (tn + fp) > 0: specificities.append(tn / (tn + fp))
    return (np.mean(specificities) * 100) if specificities else 0

# ============================================================
# DATA LOADING (CACHED)
# ============================================================
@st.cache_data(show_spinner="Loading data from repository and extracting features...")
def load_pipeline_data(selected_mode, config):
    repo_name = config['repo_name']
    data_path = config['data_path']
    folder_list = config['folders']

    # 1. Clone Repository
    if os.path.exists(repo_name):
        shutil.rmtree(repo_name)
    
    subprocess.run(["git", "clone", config['repo_url']], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not os.path.exists(data_path):
        return None, None, None, None, "Repository structure mismatch or download failed."

    # 2. Load Data
    all_features = []
    all_labels = []
    all_speeds = []
    file_stats = defaultdict(int)

    for folder_config in folder_list:
        folder_path = os.path.join(data_path, folder_config['path'])
        label = folder_config['label']
        name = folder_config['name']

        if not os.path.exists(folder_path):
            continue

        search_pattern = os.path.join(folder_path, "**", "*.csv")
        all_files = glob.glob(search_pattern, recursive=True)

        for file_path in all_files:
            try:
                current_speed = extract_speed_from_file(file_path)
                df = pd.read_csv(file_path, header=2)

                if df.shape[1] >= 2:
                    signal = df.iloc[:, 1].values
                    if not np.issubdtype(signal.dtype, np.number) or len(signal) < WINDOW_SIZE: 
                        continue

                    num_windows = (len(signal) - WINDOW_SIZE) // STRIDE + 1

                    for i in range(num_windows):
                        start = i * STRIDE
                        end = start + WINDOW_SIZE
                        window = signal[start:end]

                        # Original
                        features = extract_features(window)
                        all_features.append(features)
                        all_labels.append(label)
                        all_speeds.append(current_speed)
                        file_stats[name] += 1

                        # Augmentations
                        for aug_idx in range(AUGMENTATION_FACTOR):
                            aug_type = ['noise', 'scale', 'jitter', 'time_shift'][aug_idx % 4]
                            aug_window = augment_signal(window, aug_type)
                            aug_features = extract_features(aug_window)
                            all_features.append(aug_features)
                            all_labels.append(label)
                            all_speeds.append(current_speed)
                            file_stats[name] += 1

            except Exception as e:
                continue

    if len(all_features) == 0:
        return None, None, None, None, "No data found in CSV files."

    X = np.array(all_features)
    y = np.array(all_labels)
    s = np.array(all_speeds)
    
    # Clean NaN
    X = np.nan_to_num(X)
    
    return X, y, s, file_stats, "Success"

# ============================================================
# MAIN APP
# ============================================================

st.title("🚀 Gearbox Fault Diagnosis Dashboard")
st.sidebar.header("Configuration")

# 1. Mode Selection
selected_mode = st.sidebar.radio("Select Gearbox Type:", list(APP_CONFIG.keys()))
current_config = APP_CONFIG[selected_mode]

# 2. Data Loading Button
if st.sidebar.button("🔄 Load Data", use_container_width=True):
    with st.spinner(f"Downloading {selected_mode} data and processing..."):
        X, y, speeds, stats, msg = load_pipeline_data(selected_mode, current_config)
        st.session_state['X'] = X
        st.session_state['y'] = y
        st.session_state['speeds'] = speeds
        st.session_state['stats'] = stats
        st.session_state['loaded'] = True
        st.session_state['trained'] = False
    st.sidebar.success("Data Loaded Successfully!")

# Check if data is loaded
if 'loaded' not in st.session_state or not st.session_state['loaded']:
    st.info("👈 Please select a Gearbox Type and click 'Load Data' to begin.")
    st.stop()

X = st.session_state['X']
y = st.session_state['y']
speeds = st.session_state['speeds']
stats = st.session_state['stats']

# 3. Display Stats
st.subheader("📂 Dataset Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Samples", len(y))
col2.metric("Fault Classes", len(np.unique(y)))
col3.metric("Unique Speeds Detected", len(np.unique(speeds[speeds>0])) if len(speeds) > 0 else 0)

# Distribution Plot
st.write("**Sample Distribution by Class:**")
fig_dist, ax_dist = plt.subplots(figsize=(10, 4))
classes = list(stats.keys())
counts = list(stats.values())
colors = ['green' if 'Healthy' in c else 'red' for c in classes]
bars = ax_dist.bar(classes, counts, color=colors, alpha=0.7, edgecolor='black')
ax_dist.set_title('Sample Distribution')
ax_dist.tick_params(axis='x', rotation=45)
for bar in bars:
    height = bar.get_height()
    ax_dist.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', ha='center', va='bottom')
st.pyplot(fig_dist)
plt.clf()

# 4. Train & Analyze Button
st.divider()
if st.button("🚀 Train Models & Generate Analysis", use_container_width=True, type="primary"):
    with st.spinner("Training Random Forest, SVM, KNN, Logistic Regression, and Decision Tree... This may take a moment."):
        # --- PREPARATION ---
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            X, y, speeds, test_size=0.3, random_state=100
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # --- MODELS ---
        models = {
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
            "SVM (RBF)": SVC(kernel='rbf', random_state=42, probability=True),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "Logistic Reg": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(random_state=42)
        }

        # Fit Models
        for name, model in models.items():
            if name in ["SVM (RBF)", "KNN", "Logistic Reg"]:
                model.fit(X_train_scaled, y_train)
            else:
                model.fit(X_train, y_train)

        # Store in session state to avoid re-training on every rerender
        st.session_state['models'] = models
        st.session_state['X_test'] = X_test
        st.session_state['X_test_scaled'] = X_test_scaled
        st.session_state['y_test'] = y_test
        st.session_state['s_test'] = s_test
        st.session_state['trained'] = True
    
    st.rerun()

if 'trained' not in st.session_state or not st.session_state['trained']:
    st.info("Click 'Train Models & Generate Analysis' to view results.")
    st.stop()

# Retrieve Trained Objects
models = st.session_state['models']
X_test = st.session_state['X_test']
X_test_scaled = st.session_state['X_test_scaled']
y_test = st.session_state['y_test']
s_test = st.session_state['s_test']
class_names = current_config['class_names']

# ============================================================
# VISUALIZATION TABS
# ============================================================
st.divider()
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Learning Curve", "⚙️ Speed Accuracy", "📊 Metrics", "🎯 Per-Fault Acc", 
    "🧠 Confusion Matrix & ROC", "🌌 t-SNE Clustering", "🩺 Anomaly & Health", "🔗 Correlation"
])

# --- TAB 1: LEARNING CURVE ---
with tab1:
    st.subheader("Random Forest: Training vs Test Accuracy")
    fig_lc, ax_lc = plt.subplots(figsize=(10, 5))
    
    # Recalculate quickly for the plot (or could have cached it)
    # To save time, we will just plot the final point or re-run a quick loop
    # For demo speed, let's re-run a subset (e.g. 50 iterations) or assume user waits.
    # We will re-run the loop logic here specifically for the plot.
    rf = models["Random Forest"]
    # We can't easily get the intermediate history from sklearn without re-fitting or warm_start loop
    # We'll do a quick warm_start loop for the graph
    rf_plot = RandomForestClassifier(warm_start=True, max_depth=100, min_samples_split=5, min_samples_leaf=2, random_state=42, n_jobs=-1)
    n_iterations = 100 # Reduced for UI speed
    train_acc, test_acc = [], []
    
    with st.spinner("Generating Learning Curve data..."):
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        for i in range(1, n_iterations + 1):
            rf_plot.n_estimators = i
            rf_plot.fit(X_tr, y_tr)
            train_acc.append(accuracy_score(y_tr, rf_plot.predict(X_tr)))
            test_acc.append(accuracy_score(y_val, rf_plot.predict(X_val)))
            
    ax_lc.plot(range(1, n_iterations + 1), train_acc, 'b-', label='Training Accuracy')
    ax_lc.plot(range(1, n_iterations + 1), test_acc, 'r-', label='Validation Accuracy')
    ax_lc.set_title('Random Forest Learning Curve')
    ax_lc.set_xlabel('Iterations (Trees)')
    ax_lc.set_ylabel('Accuracy')
    ax_lc.legend()
    ax_lc.grid(True)
    st.pyplot(fig_lc)
    plt.clf()

# --- TAB 2: SPEED ACCURACY ---
with tab2:
    st.subheader("Classification Accuracy W.R.T Speed (Hz)")
    unique_speeds = np.unique(s_test)
    unique_speeds = unique_speeds[unique_speeds > 0]
    unique_speeds = np.sort(unique_speeds)

    if len(unique_speeds) > 0:
        speed_results = {name: [] for name in models}
        for speed in unique_speeds:
            idx = np.where(s_test == speed)[0]
            if len(idx) < 5: continue
            X_sub_scaled = X_test_scaled[idx]
            X_sub = X_test[idx]
            y_sub = y_test[idx]
            for name, model in models.items():
                preds = model.predict(X_sub_scaled) if name in ["SVM (RBF)", "KNN", "Logistic Reg"] else model.predict(X_sub)
                speed_results[name].append(accuracy_score(y_sub, preds) * 100)

        fig_spd, ax_spd = plt.subplots(figsize=(14, 6))
        x = np.arange(len(unique_speeds))
        width = 0.15
        multiplier = 0
        color_map = {
            "Random Forest": '#1f77b4', "SVM (RBF)": '#d62728', "KNN": '#2ca02c',
            "Logistic Reg": '#ff7f0e', "Decision Tree": '#9467bd'
        }
        for name, accuracies in speed_results.items():
            offset = width * multiplier
            rects = ax_spd.bar(x + offset, accuracies, width, label=name, color=color_map[name], edgecolor='black')
            multiplier += 1
        ax_spd.set_ylabel('Accuracy (%)')
        ax_spd.set_xlabel('Speed (Hz)')
        ax_spd.set_title('Accuracy vs Speed')
        ax_spd.set_xticks(x + width * 2)
        ax_spd.set_xticklabels([f"{int(s)} Hz" for s in unique_speeds])
        ax_spd.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax_spd.set_ylim(0, 105)
        ax_spd.grid(axis='y', linestyle='--')
        st.pyplot(fig_spd)
        plt.clf()
    else:
        st.warning("No speed information found in the test set.")

# --- TAB 3: METRICS ---
with tab3:
    st.subheader("Performance Metrics (Sensitivity, Specificity, F1)")
    metric_names = []
    sensitivity_vals = []
    specificity_vals = []
    f1_vals = []

    for name, model in models.items():
        preds = model.predict(X_test_scaled) if name in ["SVM (RBF)", "KNN", "Logistic Reg"] else model.predict(X_test)
        sens = recall_score(y_test, preds, average='weighted') * 100
        spec = calculate_specificity(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted') * 100
        metric_names.append(name)
        sensitivity_vals.append(sens)
        specificity_vals.append(spec)
        f1_vals.append(f1)

    fig_met, ax_met = plt.subplots(figsize=(12, 6))
    x = np.arange(len(metric_names))
    width = 0.25
    rects1 = ax_met.bar(x - width, sensitivity_vals, width, label='Sensitivity', color='#e15759', edgecolor='black')
    rects2 = ax_met.bar(x, specificity_vals, width, label='Specificity', color='#4e79a7', edgecolor='black')
    rects3 = ax_met.bar(x + width, f1_vals, width, label='F1-score', color='#ff9f40', edgecolor='black')
    ax_met.set_ylabel('Score (%)')
    ax_met.set_title('Model Performance Comparison')
    ax_met.set_xticks(x)
    ax_met.set_xticklabels(metric_names)
    ax_met.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_met.set_ylim(0, 105)
    ax_met.grid(axis='y', linestyle='--')
    st.pyplot(fig_met)
    plt.clf()

# --- TAB 4: PER-FAULT ACCURACY ---
with tab4:
    st.subheader("Per-Fault Classification Accuracy")
    per_model_fault_acc = {}
    for name, model in models.items():
        preds = model.predict(X_test_scaled) if name in ["SVM (RBF)", "KNN", "Logistic Reg"] else model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        with np.errstate(divide='ignore', invalid='ignore'):
            accs = (cm.diagonal() / cm.sum(axis=1)) * 100
            accs[np.isnan(accs)] = 0
        per_model_fault_acc[name] = accs

    # Filter out healthy class (index 0) for per-fault graph if desired, or keep all
    # Keeping all for consistency
    fault_names = class_names 
    
    fig_pfa, ax_pfa = plt.subplots(figsize=(14, 7))
    x = np.arange(len(fault_names))
    width = 0.15
    multiplier = 0
    for name, data in per_model_fault_acc.items():
        offset = width * multiplier
        rects = ax_pfa.bar(x + offset, data, width, label=name, color=color_map[name], edgecolor='black')
        multiplier += 1
    ax_pfa.set_ylabel('Accuracy (%)')
    ax_pfa.set_xlabel('Fault Type')
    ax_pfa.set_title('Accuracy by Fault Class')
    ax_pfa.set_xticks(x + width * 2)
    ax_pfa.set_xticklabels(fault_names, rotation=45, ha='right')
    ax_pfa.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax_pfa.set_ylim(0, 105)
    ax_pfa.grid(axis='y', linestyle='--')
    st.pyplot(fig_pfa)
    plt.clf()

# --- TAB 5: CONFUSION MATRIX & ROC ---
with tab5:
    col_a, col_b = st.columns(2)
    
    # Confusion Matrix
    with col_a:
        st.subheader("Confusion Matrix (Random Forest)")
        rf = models["Random Forest"]
        y_pred_rf = rf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_rf)
        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
        ax_cm.set_title('Confusion Matrix')
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        st.pyplot(fig_cm)
        plt.clf()
        
        st.text("Classification Report (RF):")
        st.text(classification_report(y_test, y_pred_rf, target_names=class_names))

    # ROC
    with col_b:
        st.subheader("ROC Curves (One-vs-All)")
        y_test_bin = label_binarize(y_test, classes=list(range(len(class_names))))
        n_classes = y_test_bin.shape[1]
        y_score = rf.predict_proba(X_test)

        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        cmap = plt.get_cmap('tab10')
        colors = cmap(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            ax_roc.plot(fpr[i], tpr[i], color=color, lw=2,
                        label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=2)
        ax_roc.set_xlim([0.0, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC Curves')
        ax_roc.legend(loc="lower right", fontsize='small')
        st.pyplot(fig_roc)
        plt.clf()

# --- TAB 6: t-SNE ---
with tab6:
    st.subheader("Feature Clustering (t-SNE)")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    with st.spinner("Running t-SNE..."):
        X_embedded = tsne.fit_transform(X_test)
    
    fig_tsne, ax_tsne = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap('tab10')
    for i in range(len(class_names)):
        indices = np.where(y_test == i)
        ax_tsne.scatter(X_embedded[indices, 0], X_embedded[indices, 1],
                        c=[cmap(i)], label=class_names[i], alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    ax_tsne.set_title('t-SNE Feature Clustering')
    ax_tsne.set_xlabel('Component 1')
    ax_tsne.set_ylabel('Component 2')
    ax_tsne.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot(fig_tsne)
    plt.clf()

# --- TAB 7: ANOMALY & HEALTH ---
with tab7:
    col_c, col_d = st.columns(2)
    
    rf = models["Random Forest"]
    probs = rf.predict_proba(X_test)
    health_scores = probs[:, 0] # Probability of Class 0 (Healthy)
    anomaly_scores = 1 - health_scores
    
    with col_c:
        st.subheader("Anomaly Detection Scores")
        sorted_indices = np.argsort(anomaly_scores)
        sorted_scores = anomaly_scores[sorted_indices]
        
        fig_anom, ax_anom = plt.subplots(figsize=(10, 5))
        colors = ['green' if score < 0.5 else 'red' for score in sorted_scores]
        ax_anom.bar(range(len(sorted_scores)), sorted_scores, color=colors, alpha=0.7)
        ax_anom.axhline(0.5, color='black', linestyle='--', label='Threshold')
        ax_anom.set_title('Anomaly Score Distribution (Sorted)')
        ax_anom.set_ylabel('Anomaly Score (1 - P(Healthy))')
        ax_anom.set_xlabel('Sample Index (Sorted)')
        ax_anom.legend()
        st.pyplot(fig_anom)
        plt.clf()

    with col_d:
        st.subheader("Health Index Analysis")
        healthy_idx = np.where(y_test == 0)[0]
        faulty_idx = np.where(y_test != 0)[0]
        
        fig_hi, ax_hi = plt.subplots(figsize=(10, 5))
        ax_hi.scatter(range(len(health_scores)), health_scores, c=health_scores, cmap='RdYlGn', alpha=0.6)
        ax_hi.axhline(0.5, color='black', linestyle='--')
        ax_hi.set_title('Health Index (0=Faulty, 1=Healthy)')
        ax_hi.set_ylabel('Probability of being Healthy')
        ax_hi.set_xlabel('Test Sample')
        ax_hi.set_ylim(0, 1.05)
        st.pyplot(fig_hi)
        plt.clf()
        
        st.metric("Avg Health (Healthy Class)", f"{np.mean(health_scores[healthy_idx]):.3f}" if len(healthy_idx)>0 else "N/A")
        st.metric("Avg Health (Faulty Classes)", f"{np.mean(health_scores[faulty_idx]):.3f}" if len(faulty_idx)>0 else "N/A")

# --- TAB 8: CORRELATION ---
with tab8:
    st.subheader("Feature Correlation Heatmap")
    feature_names = ['Mean', 'Std', 'Max', 'Min', 'RMS', 'Skew', 'Kurtosis',
                     'MAD', 'Energy', 'Dom_Freq', 'Spec_Energy', 'Spec_Std']
    df_features = pd.DataFrame(X, columns=feature_names)
    
    fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
    sns.heatmap(df_features.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr)
    ax_corr.set_title('Feature Correlation Matrix')
    st.pyplot(fig_corr)
    plt.clf()
