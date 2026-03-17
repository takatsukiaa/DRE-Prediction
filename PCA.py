import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns  # 加入 seaborn 讓圖表更美觀
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from collections import defaultdict
import tkinter as tk
from tkinter import filedialog
from sklearn.metrics import adjusted_rand_score, cluster

# 設定不限制顯示的列數 (Rows)
pd.set_option('display.max_rows', None)
# 設定不限制欄位寬度 (避免內容太長被切掉)
pd.set_option('display.max_colwidth', None)
# 設定顯示所有欄位 (Columns)
pd.set_option('display.max_columns', None)

DPI = 600
# (1. 藥物與 DDD 標準配置部分保持不變...)
DRUG_MAP = {
    "Lamictal 50mg/tab": ("Lamotrigine", 50), "Lyrica 75mg/cap": ("Pregabalin", 75),
    "Topamax 100mg/tab": ("Topiramate", 100), "Fycompa 2mg/tab": ("Perampanel", 2),
    "Keppra 500mg/tab": ("Levetiracetam", 500), "Trileptal 300mg/tab": ("Oxcarbazepine", 300),
    "Depakine 500mg/tab": ("Valproate", 500), "Frisium 10mg/tab": ("Clobazam", 10),
    "Vimpat 100mg/tab": ("Lacosamide", 100), "Zonegran 100mg/tab": ("Zonisamide", 100),
    "Carpine 200mg/tab": ("Carbamazepine", 200), "Aclonax 0.5mg/tab": ("Clonazepam", 0.5),
    "Rivotril 0.5mg/tab": ("Clonazepam", 0.5), "Carbamazepine 200mg/tab": ("Carbamazepine", 200),
    "Tegretol CR 200mg/tab": ("Carbamazepine", 200), "Dilantin 100mg/cap": ("Phenytoin", 100),
    "Phenytoin 100mg/cap": ("Phenytoin", 100), "Fycompa 2mg": ("Perampanel", 2),
    "Topiramate 25mg/cap": ("Topiramate", 25), "Topiramate 100mg/tab": ("Topiramate", 100),
    "Phenobarbital 30mg/tab": ("Phenobarbital", 30), "Depakine 200mg/mL oral sol": ("Valproate", 200),
    "Depakine Chrono 500mg/tab": ("Valproate", 500), "Depakine Chrono 200mg/tab": ("Valproate", 200),
    "Risperdal 1 mg/tab": ("Risperidone", 1)
}

DDD_TABLE = {
    "Lamotrigine": 300, "Pregabalin": 300, "Topiramate": 300, "Perampanel": 8,
    "Levetiracetam": 1500, "Oxcarbazepine": 1000, "Carbamazepine": 1000,
    "Valproate": 1500, "Clobazam": 20, "Lacosamide": 300, "Zonisamide": 200,
    "Clonazepam": 8, "Phenytoin": 300, "Phenobarbital": 100, "Risperidone": 5
}

ground_truth = {
    '2': 0, '21': 0, '29': 0, '50':0,'56':0,'63':0,'86':0,'109':0,'117':0,'136':0,'140':0,'156':0,'157':0,'158':0,'159':0,
    '3':1,'24':1,'27':1,'39':1,'41':1,'42':1,'65':1,'69':1,'74':1,'95':1,'102':1,'139':1,'160':1,'161':1,'162':1,
    # 1 = DRE 0 = NDRE
}
# (2. 特徵提取函數，加入藥物種類計數)
def extract_robust_features(file_path):
    try:
        df = pd.read_csv(file_path)
        df.iloc[:, 0] = df.iloc[:, 0].astype(str).str.strip()
        pid = str(df.iloc[0, 1]).strip()
        target_keys = {
                        #'seizure': 'Seizure',
                       'spike': 'Spike ED',
                       'sharp': 'Sharp ED',
                       'delta': 'Delta waves',
                       'theta': 'Theta waves',
                       #'onset': 'Age of onset',
                       }
        res = {'id': pid, 'series': {}}
        for key, kw in target_keys.items():
            mask = df.iloc[:, 0].str.contains(kw, case=False, na=False, regex=False)
            if mask.any():
                vals = pd.to_numeric(df[mask].iloc[0, 1:], errors='coerce')
                res['series'][key] = vals.tolist()
            else:
                res['series'][key] = []

        # total_months = df.shape[1] - 1
        # ddd_scores = np.zeros(total_months)
        # drug_count_scores = np.zeros(total_months)  # 新增：記錄每個月吃幾種藥

        # for i in range(len(df)):
        #     name = df.iloc[i, 0]
        #     if name in DRUG_MAP:
        #         comp, strength = DRUG_MAP[name]
        #         ddd_std = DDD_TABLE.get(comp, 1.0)
        #         counts = pd.to_numeric(df.iloc[i, 1:], errors='coerce').fillna(0).values
        #
        #         # 計算藥物 DDD 負荷
        #         ddd_scores += (counts * strength) / ddd_std
        #
        #         # 新增邏輯：只要當月該藥物服用數量大於 0，藥物種類數就 +1
        #         drug_count_scores += (counts > 0).astype(int)
        #
        # res['series']['ddd_load'] = ddd_scores.tolist()
        # res['series']['drug_count'] = drug_count_scores.tolist()  # 將用藥數量存入 series

        return res
    except Exception as e:
        print(f"檔案 {file_path} 處理失敗: {e}")
        return None

def calculate_cohens_d(group0, group1):
    # 計算平均值與樣本數
    n1, n2 = len(group0), len(group1)
    m1, m2 = np.mean(group0), np.mean(group1)

    # 計算變異數 (Variance)
    v1, v2 = np.var(group0, ddof=1), np.var(group1, ddof=1)

    # 計算合併標準差 (Pooled Standard Deviation)
    pooled_sd = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))

    # 回傳 Cohen's d (取絕對值代表差異強度)
    return abs(m1 - m2) / pooled_sd

# ==========================================
# 3. 主分析流程
# ==========================================
def run_refined_analysis(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "Pt*.csv"))
    if not csv_files:
        print(f"在路徑 '{folder_path}' 找不到以 Pt 開頭的 CSV 檔案。")
        return

    patient_vault = defaultdict(lambda: defaultdict(list))
    for f in csv_files:
        data = extract_robust_features(f)
        if data:
            pid = data['id']
            for k, series in data['series'].items():
                patient_vault[pid][k].extend(series)

    final_rows = []
    for pid, s_dict in patient_vault.items():
        row = {'Patient_ID': pid}
        for key, vals in s_dict.items():
            if vals:
                arr = pd.Series(vals)
                row[f'{key}_mean'] = arr.mean()
            else:
                row[f'{key}_mean'] = np.nan
        final_rows.append(row)

    df_final = pd.DataFrame(final_rows)

    # 數據補值與標準化
    X_raw = df_final.drop(columns=['Patient_ID'])
    imputer = KNNImputer(n_neighbors=3)
    X_imputed = imputer.fit_transform(X_raw)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    # 分群 (您可以自行修改 n_clusters，目前為 2)
    n_clusters = 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df_final['Cluster'] = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, df_final['Cluster'])
    print(f"分群穩定性指標 (Silhouette Score): {score:.3f}")
    # --- 輸出 Pandas 內建摘要 ---
    print("\n" + "=" * 80)
    print("【 1. 全體病患數據敘述統計 (Pandas describe) 】")
    print("=" * 80)
    print(df_final.drop(columns='Cluster').describe().round(2))

    print("\n" + "=" * 80)
    print("【 2. 各群組特徵平均值對比 (Groupby Mean) 】")
    print("=" * 80)
    # 這是最直觀的分群特性表，現在會包含 drug_count_mean
    print(df_final.groupby('Cluster').mean(numeric_only=True).round(2))

    print("\n" + "=" * 80)
    print("【 3. 各群組成員清單 】")
    print("=" * 80)
    members = df_final.groupby('Cluster')['Patient_ID'].apply(list)
    for cid, plist in members.items():
        print(f"群組 {cid} (共 {len(plist)} 人): {plist}")
    # --- 計算 ARI 與外部驗證 ---

    # A. 將 ground_truth 字典對應到 DataFrame 中
    # 確保 Patient_ID 與 ground_truth 的 Key 類型一致 (都是字串)
    df_final['True_Label'] = df_final['Patient_ID'].map(ground_truth)

    # B. 由於 ground_truth 可能不包含所有病患，我們只針對有標籤的樣本進行評估
    labeled_df = df_final.dropna(subset=['True_Label'])

    if not labeled_df.empty:
        y_true = labeled_df['True_Label']
        y_pred = labeled_df['Cluster']

        # 計算 ARI
        ari_value = adjusted_rand_score(y_true, y_pred)

        print("\n" + "=" * 80)
        print(f"【 外部驗證指標：Adjusted Rand Index (ARI) 】")
        print("=" * 80)
        print(f"ARI Score: {ari_value:.4f}")
        print("註：1.0 為完美匹配，0.0 為隨機分群，負值則比隨機更差。")

        # C. 建立列聯矩陣 (Contingency Matrix) 觀察分佈
        # 這可以讓您看到真實標籤與分群編號的交叉分佈
        conf_matrix = pd.crosstab(y_true, y_pred, rownames=['True'], colnames=['Cluster'])
        print("\n列聯矩陣 (Contingency Matrix):")
        print(conf_matrix)
    else:
        print("\n警告：找不到符合 ground_truth 的病患 ID，無法計算 ARI。")

    pca = PCA(n_components=3)
    pca_res = pca.fit_transform(X_scaled)
    ev = pca.explained_variance_ratio_ * 100
    # === 新增：PCA 二維視覺化 (Score Plot) ===
    plt.figure(figsize=(10, 7), dpi=DPI)

    # 建立一個 DataFrame 方便 seaborn 繪圖
    pca_df = pd.DataFrame(
        data=pca_res[:, :2],
        columns=['PC1', 'PC2']
    )
    pca_df['Cluster'] = df_final['Cluster'].values
    pca_df['Patient_ID'] = df_final['Patient_ID'].values

    # 繪製散佈圖
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='Cluster',
        palette='viridis',
        data=pca_df,
        s=100, alpha=0.8, edgecolor='w'
    )

    # 在點位旁邊標註病患 ID (可選)
    for i in range(pca_df.shape[0]):
        plt.text(
            pca_df.PC1[i] + 0.02, pca_df.PC2[i] + 0.02,
            pca_df.Patient_ID[i],
            fontsize=9, alpha=0.7
        )

    plt.title('PCA Score Plot (PC1 vs PC2)', fontsize=15)
    plt.xlabel(f'PC1 ({ev[0]:.1f}% Variance Explained)', fontsize=12)
    plt.ylabel(f'PC2 ({ev[1]:.1f}% Variance Explained)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Cluster')

    # 儲存圖表
    plt.savefig('PCA_Score_Plot.png')
    print("PCA 二維散佈圖已儲存為 PCA_Score_Plot.png")

    print(f'PC1 (Explained Variance: {ev[0]:.1f}%)')
    print(f'PC2 (Explained Variance: {ev[1]:.1f}%)')
    print(f'PC3 (Explained Variance: {ev[2]:.1f}%)')
    # print(f'PC4 (Explained Variance: {ev[3]:.1f}%)')

    all_mean_cols = [col for col in df_final.columns if col.endswith('_mean')]
    print(f"本次比較的特徵包含：{', '.join(all_mean_cols)}")



    loadings = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2','PC3',],
        index=X_raw.columns
    )
    print("\nLoadings")
    print(loadings.abs().sort_values(by='PC1', ascending=False))


from scipy.stats import mannwhitneyu
def calculate_feature_significance(df):
    features = [col for col in df.columns if col.endswith('_mean')]
    results = []

    for feat in features:
        # 分出兩群的數據
        group0 = df[df['Cluster'] == 0][feat].dropna()
        group1 = df[df['Cluster'] == 1][feat].dropna()

        # 使用 Mann-Whitney U 檢定 (適合樣本數小且不一定符合常態分佈的臨床數據)
        stat, p = mannwhitneyu(group0, group1)

        results.append({'Feature': feat, 'P-value': p, 'Significance': '*' if p < 0.05 else 'NS'})

    return pd.DataFrame(results)











# ==========================================
# 4. 執行進入點
# ==========================================
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    selected_path = filedialog.askdirectory(title="選擇病人資料夾")
    if selected_path:
        run_refined_analysis(selected_path)