# =====================================================
# AI岗位市场数据分析（基于 ai_job_dataset1.csv）
# 清洗 + 可视化(保存PNG) + 回归建模(5折CV + GridSearch + 模型对比)
# =====================================================

import os
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


# ============ 工具函数 ============
def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def save_fig(filename: str, out_dir: str = "outputs", dpi: int = 300):
    ensure_dir(out_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, filename), dpi=dpi)
    plt.close()


def safe_split_skills(x):
    """把 'Python, SQL, Docker' -> ['python','sql','docker']"""
    if not isinstance(x, str) or x.strip() == "":
        return []
    parts = [p.strip().lower() for p in x.split(",")]
    parts = [p for p in parts if p]
    return parts


def clip_outliers(series: pd.Series, low_q=0.01, high_q=0.99):
    """用分位数截断异常值，避免删太多数据"""
    low = series.quantile(low_q)
    high = series.quantile(high_q)
    return series.clip(low, high)


# ============ 1) 数据读取 & 基本信息 ============
INPUT_CSV = "ai_job_dataset1.csv"
OUT_DIR = "outputs"
ensure_dir(OUT_DIR)

df = pd.read_csv(INPUT_CSV)

print("【原始数据】shape:", df.shape)
print(df.head())
print(df.info())


# ============ 2) 数据清洗（重复 / 缺失 / 异常） ============
# 2.1 重复值检测及处理
dup_rows = df.duplicated().sum()
dup_jobid = df.duplicated(subset=["job_id"]).sum() if "job_id" in df.columns else None
print(f"\n【重复值】整行重复: {dup_rows} 行；job_id 重复: {dup_jobid} 行")

# 优先按 job_id 去重（如果 job_id 本来就是唯一，这里不会删）
if "job_id" in df.columns:
    df = df.drop_duplicates(subset=["job_id"])
else:
    df = df.drop_duplicates()

print("去重后 shape:", df.shape)

# 2.2 缺失值检测及处理
missing = df.isna().sum().sort_values(ascending=False)
print("\n【缺失值统计】前10列：")
print(missing.head(10))

# 2.3 异常值检测及处理（这里重点处理 salary_usd）
# salary_usd 应该为正数
df["salary_usd"] = pd.to_numeric(df["salary_usd"], errors="coerce")
before = df.shape[0]
df = df.dropna(subset=["salary_usd"])
df = df[df["salary_usd"] > 0]
after = df.shape[0]
print(f"\n【薪资异常处理】删除 salary_usd 缺失或<=0: {before - after} 行")

# 对薪资做分位数截断（保留数据规模，降低极端值影响）
df["salary_usd"] = clip_outliers(df["salary_usd"], 0.01, 0.99)

# ============ 3) 特征工程（论文第3章预处理） ============
# 3.1 时间字段处理：posting_date / application_deadline
for col in ["posting_date", "application_deadline"]:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# deadline_days: 截止日期-发布日期（天数）
df["deadline_days"] = (df["application_deadline"] - df["posting_date"]).dt.days

# posting_month
df["posting_month"] = df["posting_date"].dt.month

# 3.2 技能数量
df["skill_list"] = df["required_skills"].apply(safe_split_skills)
df["skill_count"] = df["skill_list"].apply(len)

# 3.3 数值列规范
numeric_cols = [
    "salary_usd", "years_experience", "remote_ratio",
    "job_description_length", "benefits_score", "deadline_days", "posting_month", "skill_count"
]
for c in numeric_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# 3.4 缺失值填补（建模需要：数值用中位数，类别用 Unknown）
num_fill_cols = ["years_experience", "remote_ratio", "job_description_length", "benefits_score", "deadline_days", "posting_month", "skill_count"]
for c in num_fill_cols:
    if c in df.columns:
        df[c] = df[c].fillna(df[c].median())

cat_fill_cols = [
    "job_title", "salary_currency", "experience_level", "employment_type",
    "company_location", "company_size", "employee_residence",
    "education_required", "industry", "company_name"
]
for c in cat_fill_cols:
    if c in df.columns:
        df[c] = df[c].fillna("Unknown").astype(str)

# 保存清洗后数据
clean_path = os.path.join(OUT_DIR, "ai_job_dataset1_clean.csv")
df.to_csv(clean_path, index=False, encoding="utf-8-sig")
print("\n✅ 清洗后数据保存：", clean_path)
print("【清洗后数据】shape:", df.shape)


# ============ 4) 数据分析与可视化（保存PNG，≥6张） ============
# 4.1.1 数据摘要信息（你报告里可以写：字段/类型/缺失情况/规模）
summary_path = os.path.join(OUT_DIR, "data_summary.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("【数据规模】\n")
    f.write(str(df.shape) + "\n\n")
    f.write("【字段信息】\n")
    f.write(str(df.dtypes) + "\n\n")
    f.write("【缺失值统计】\n")
    f.write(str(df.isna().sum()) + "\n\n")
    f.write("【数值字段描述性统计】\n")
    f.write(str(df[numeric_cols].describe()) + "\n")
print("✅ 数据摘要信息保存：", summary_path)

# 4.1.2 描述性统计分析（打印即可，也写入文件）
desc = df[numeric_cols].describe()
print("\n【描述性统计】\n", desc)

# ---- 图1：薪资分布直方图（分布类）----
plt.figure(figsize=(7, 4))
plt.hist(df["salary_usd"], bins=40, alpha=0.85, edgecolor="white")
plt.xlabel("薪资（USD）")
plt.ylabel("岗位数量")
plt.title("图4-1 薪资分布直方图")
save_fig("图4-1_薪资分布直方图.png", OUT_DIR)

# ---- 图2：薪资箱线图（分布类）----
plt.figure(figsize=(6, 4))
sns.boxplot(y=df["salary_usd"])
plt.title("图4-2 薪资箱线图")
save_fig("图4-2_薪资箱线图.png", OUT_DIR)

# ---- 图3：相关性热力图（关联类）----
corr_cols = ["salary_usd", "years_experience", "remote_ratio", "job_description_length", "benefits_score", "deadline_days", "skill_count"]
corr = df[corr_cols].corr()
plt.figure(figsize=(7, 5))
sns.heatmap(corr, annot=True, fmt=".2f")
plt.title("图4-3 数值变量相关性热力图")
save_fig("图4-3_相关性热力图.png", OUT_DIR)

# ---- 图4：技能数量 vs 薪资（关联类）----
plt.figure(figsize=(7, 4))
plt.scatter(df["skill_count"], df["salary_usd"], alpha=0.25)
plt.xlabel("技能数量（skill_count）")
plt.ylabel("薪资（USD）")
plt.title("图4-4 技能数量与薪资关系散点图")
save_fig("图4-4_技能数量与薪资散点图.png", OUT_DIR)

# ---- 图5：经验等级平均薪资（对比类）----
plt.figure(figsize=(8, 4))
tmp = df.groupby("experience_level")["salary_usd"].mean().sort_values(ascending=False)
tmp.plot(kind="bar")
plt.ylabel("平均薪资（USD）")
plt.title("图4-5 不同经验等级的平均薪资对比")
save_fig("图4-5_经验等级平均薪资.png", OUT_DIR)

# ---- 图6：行业岗位数量Top10（对比类）----
plt.figure(figsize=(9, 4))
top_ind = df["industry"].value_counts().head(10)
top_ind.plot(kind="bar")
plt.ylabel("岗位数量")
plt.title("图4-6 行业岗位数量Top10")
save_fig("图4-6_行业岗位数量Top10.png", OUT_DIR)

# ---- 图7：行业平均薪资Top10（对比类）----
plt.figure(figsize=(9, 4))
ind_salary = df.groupby("industry")["salary_usd"].mean().sort_values(ascending=False).head(10)
ind_salary.plot(kind="bar")
plt.ylabel("平均薪资（USD）")
plt.title("图4-7 行业平均薪资Top10")
save_fig("图4-7_行业平均薪资Top10.png", OUT_DIR)

# ---- 图8：远程比例 vs 薪资箱线图（对比类）----
plt.figure(figsize=(7, 4))
sns.boxplot(x=df["remote_ratio"].astype(int), y=df["salary_usd"])
plt.xlabel("远程比例 remote_ratio（0/50/100）")
plt.ylabel("薪资（USD）")
plt.title("图4-8 远程比例与薪资分布（箱线图）")
save_fig("图4-8_远程比例与薪资箱线图.png", OUT_DIR)

# ---- 技能频次统计（用于技能需求可视化）----
all_skills = df["skill_list"].explode()
skill_counts = all_skills.value_counts().dropna()
skill_counts.head(30).to_csv(os.path.join(OUT_DIR, "top_skills.csv"), encoding="utf-8-sig")

# ---- 图9：技能频次Top15（对比/分布类）----
plt.figure(figsize=(9, 5))
skill_counts.head(15).sort_values().plot(kind="barh")
plt.xlabel("出现次数")
plt.title("图4-9 技能需求Top15（频次）")
save_fig("图4-9_技能需求Top15.png", OUT_DIR)

print("\n✅ 可视化已保存到 outputs/ 文件夹（已满足 ≥6 张图）")


# ============ 5) 建模：回归（7:3划分 + 5折CV + GridSearch + 模型对比） ============
# 目标：预测 salary_usd

target = "salary_usd"

# 特征：数值 + 类别
num_features = ["years_experience", "remote_ratio", "job_description_length", "benefits_score", "deadline_days", "posting_month", "skill_count"]
cat_features = ["experience_level", "employment_type", "company_location", "company_size", "education_required", "industry"]

X = df[num_features + cat_features].copy()
y = df[target].copy()

# 7:3 划分 + 随机种子
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# 预处理：数值标准化 + 类别OneHot
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ],
    remainder="drop"
)

# 多个模型对比（满足“对比不同模型性能差异”）
models = {
    "Ridge": Ridge(random_state=42),
    "RandomForest": RandomForestRegressor(random_state=42),
    "GBDT": GradientBoostingRegressor(random_state=42),
}

param_grids = {
    "Ridge": {
        "model__alpha": [0.1, 1.0, 10.0, 50.0]
    },
    "RandomForest": {
        "model__n_estimators": [200, 400],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    },
    "GBDT": {
        "model__n_estimators": [200, 400],
        "model__learning_rate": [0.05, 0.1],
        "model__max_depth": [2, 3]
    }
}

results = []
best_overall = None

for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])

    grid = GridSearchCV(
        pipe,
        param_grid=param_grids[name],
        cv=5,                 # 5折交叉验证
        scoring="r2",         # 回归常用R²
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # 测试集评估
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    results.append({
        "model": name,
        "best_params": str(grid.best_params_),
        "cv_best_r2": grid.best_score_,
        "test_r2": r2,
        "test_rmse": rmse,
        "test_mae": mae
    })

    if best_overall is None or r2 > best_overall["test_r2"]:
        best_overall = {
            "name": name,
            "estimator": best_model,
            "test_r2": r2,
            "test_rmse": rmse,
            "test_mae": mae,
            "best_params": grid.best_params_,
            "y_pred": y_pred
        }

# 保存模型对比结果
result_df = pd.DataFrame(results).sort_values(by="test_r2", ascending=False)
result_df.to_csv(os.path.join(OUT_DIR, "model_compare.csv"), index=False, encoding="utf-8-sig")
print("\n✅ 模型对比结果已保存 outputs/model_compare.csv")
print(result_df)

print("\n【最佳模型】", best_overall["name"])
print("最佳参数：", best_overall["best_params"])
print("测试集 R²:", best_overall["test_r2"])
print("测试集 RMSE:", best_overall["test_rmse"])
print("测试集 MAE:", best_overall["test_mae"])

# ---- 回归评估可视化：真实vs预测 ----
plt.figure(figsize=(6, 6))
plt.scatter(y_test, best_overall["y_pred"], alpha=0.25)
minv = min(y_test.min(), best_overall["y_pred"].min())
maxv = max(y_test.max(), best_overall["y_pred"].max())
plt.plot([minv, maxv], [minv, maxv], linestyle="--")
plt.xlabel("真实薪资（USD）")
plt.ylabel("预测薪资（USD）")
plt.title(f"图5-1 最佳模型预测效果（{best_overall['name']}）")
save_fig("图5-1_best_model_prediction_vs_true.png", OUT_DIR)

# ---- 残差分布 ----
residuals = y_test.values - best_overall["y_pred"]
plt.figure(figsize=(7, 4))
plt.hist(residuals, bins=40, alpha=0.85, edgecolor="white")
plt.xlabel("残差（真实-预测）")
plt.ylabel("频数")
plt.title("图5-2 残差分布直方图")
save_fig("图5-2_best_model_residuals.png", OUT_DIR)

print("\n✅ 建模与评估可视化已保存到 outputs/")

print("\n全部完成 ✅ 你现在可以直接用 outputs/ 的图和表写报告/做答辩。")
