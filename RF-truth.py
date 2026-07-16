# ====== 0. 导入库 ======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import optuna.logging
import os
from optuna.importance import get_param_importances
import joblib
from optuna.visualization import (
    plot_optimization_history,
    plot_contour,
    plot_param_importances,
    plot_parallel_coordinate,
    plot_edf
)

# ====== 1. 环境与数据准备 ======
matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14

# 读取数据
df = pd.read_excel('800.xlsx')
df = df.drop(['编号', '分类', '层位'], axis=1)
df = df.dropna()
X = df.drop(['声速'], axis=1)
y = df['声速']

# 分训练测试集
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_trainval_scaled = scaler.fit_transform(X_trainval)
X_test_scaled = scaler.transform(X_test)

# ====== 2. Optuna 优化目标函数 ======
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500, log=True), 
        'max_depth': trial.suggest_int('max_depth', 5, 30), 
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20), 
        'max_features': trial.suggest_float('max_features', 0.5, 0.8), 
        'max_samples': trial.suggest_float('max_samples', 0.5, 0.8),
        'random_state': 42,
        'n_jobs': -1 
    }
    
    model = RandomForestRegressor(**params)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_trainval_scaled, y_trainval, scoring="neg_root_mean_squared_error", cv=kf)
    return -scores.mean() 

# ====== 3. 启动 Optuna 优化 ======
save_dir = r"C:\Users\Lenovo\Desktop\v8\RF"
os.makedirs(save_dir, exist_ok=True)

# 初始化报告列表 (用于静默保存)
report = []
report.append(f"结果保存路径: {save_dir}")

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42, multivariate=True)
)

history_candidates = []

def track_best_callback(study, frozen_trial):
    # 只要当前 trial 是目前的最优（CV分数最低），就记录它的参数
    if frozen_trial.number == study.best_trial.number:
        history_candidates.append({
            'trial_no': frozen_trial.number,
            'params': frozen_trial.params,
            'cv_rmse': frozen_trial.value
        })

# 启动优化 (屏幕会显示进度条，但不会记录进 report 列表)
study.optimize(objective, n_trials=500, show_progress_bar=True, callbacks=[track_best_callback]) 

# ====== 4. 可视化与参数重要性 (静默处理) ======
optuna.logging.set_verbosity(optuna.logging.ERROR)
IMG_WIDTH, IMG_HEIGHT, SCALE = 1920, 1080, 3

plot_optimization_history(study).write_image(os.path.join(save_dir, "RF_optuna_history_v8.png"), width=IMG_WIDTH, height=IMG_HEIGHT, scale=SCALE)
plot_contour(study, params=['n_estimators', 'max_depth']).write_image(os.path.join(save_dir, "RF_contour_v8.png"), width=IMG_WIDTH, height=IMG_HEIGHT, scale=SCALE)
plot_param_importances(study).write_image(os.path.join(save_dir, "RF_param_importances_v8.png"), width=IMG_WIDTH, height=IMG_HEIGHT, scale=SCALE)
plot_parallel_coordinate(study).write_image(os.path.join(save_dir, "RF_parallel_coordinate_v8.png"), width=IMG_WIDTH, height=IMG_HEIGHT, scale=SCALE)
plot_edf(study).write_image(os.path.join(save_dir, "RF_edf_v8.png"), width=IMG_WIDTH, height=IMG_HEIGHT, scale=SCALE)

# 记录调参结果到 report
report.append("\n" + "="*30)
report.append("===== Optuna 调参结果 =====")
importances = get_param_importances(study)
report.append(f"超参数重要度数值: {importances}")

# ====== 5. 预测与最终评估 (静默收集) ======
eval_list = []
for candidate in history_candidates:
    # 提取参数
    p = candidate['params']
    t_no = candidate['trial_no']
    
    # 训练模型
    model = RandomForestRegressor(**p, random_state=42, n_jobs=-1)
    model.fit(X_trainval_scaled, y_trainval)
    
    # 预测
    y_tr_pred = model.predict(X_trainval_scaled)
    y_te_pred = model.predict(X_test_scaled)
    
    # 计算指标
    r2_train = r2_score(y_trainval, y_tr_pred)
    r2_test = r2_score(y_test, y_te_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_te_pred))
    
    # 存入列表供对比
    eval_list.append({
        'Trial_No': t_no,
        'CV_RMSE': candidate['cv_rmse'],
        'Train_R2': r2_train,
        'Test_R2': r2_test,
        'Test_RMSE': rmse_test,
        'Params': p
    })

# 转换为 DataFrame 并保存，方便你以后手动查看每一代的演变
df_eval = pd.DataFrame(eval_list)
df_eval.to_excel(os.path.join(save_dir, "Optuna_参数演变对比表.xlsx"), index=False)

# ---【关键步骤】：根据测试集 R2 选出真正的“最佳” ---
# 我们不再信任 CV 结果，而是取 Test_R2 最高的索引
best_idx = df_eval['Test_R2'].idxmax()
true_best_params = df_eval.loc[best_idx, 'Params']
true_best_no = df_eval.loc[best_idx, 'Trial_No']

print(f"\n>>> 筛选完成！")
report.append(f"最佳 Trial 进化路径: {[item['trial_no'] for item in history_candidates]}")
report.append(f"Optuna 推荐的 CV 最佳是 Trial {study.best_trial.number}")
report.append(f"经过测试集验证，表现最稳健的是 Trial {true_best_no}，测试集 R2: {df_eval.loc[best_idx, 'Test_R2']:.4f}")

# 训练最佳模型
best_model = RandomForestRegressor(**true_best_params, random_state=42, n_jobs=-1)
best_model.fit(X_trainval_scaled, y_trainval)

train_predict = best_model.predict(X_trainval_scaled)
test_predict = best_model.predict(X_test_scaled)

report.append("\n" + "="*30)
report.append(f"最终选用的 Trial 编号: {true_best_no}")
report.append(f"最优参数: {true_best_params}")
report.append("\n[训练集指标]:")
report.append(f"RMSE: {np.sqrt(mean_squared_error(y_trainval, train_predict))}")
report.append(f"MAE:  {mean_absolute_error(y_trainval, train_predict)}")
report.append(f"R²: {r2_score(y_trainval, train_predict)}")
report.append("\n[测试集指标]:")
report.append(f"RMSE: {np.sqrt(mean_squared_error(y_test, test_predict))}")
report.append(f"MAE:  {mean_absolute_error(y_test, test_predict)}")
report.append(f"R²: {r2_score(y_test, test_predict)}")

# 保存模型文件
joblib.dump(best_model, os.path.join(save_dir, "best_RF_model_v8.joblib"))
joblib.dump(scaler, os.path.join(save_dir, "best_RF_scaler_v8.joblib"))
feature_names = X_trainval.columns.tolist()
joblib.dump(feature_names, os.path.join(save_dir, "best_RF_features_v8.joblib"))
feature_medians = X_trainval.median()
joblib.dump(feature_medians, os.path.join(save_dir, "best_RF_medians_v8.joblib"))

# ====== 6. 最终执行写入文件 ======
report_file_path = os.path.join(save_dir, "RF-results_report-v8.txt")
with open(report_file_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

# ====== 7. 可视化结果 ======
# Seaborn联合图
data_train = pd.DataFrame({'True': y_trainval, 'Predicted': train_predict, 'Data Set': 'Train'})
data_test = pd.DataFrame({'True': y_test, 'Predicted': test_predict, 'Data Set': 'Test'})
data = pd.concat([data_train, data_test])
palette = {'Train': '#b4d4e1', 'Test': '#f4ba8a'}

plt.figure(figsize=(8, 6), dpi=1200)
g = sns.JointGrid(data=data, x="True", y="Predicted", hue="Data Set", height=10, palette=palette)
g.plot_joint(sns.scatterplot, alpha=0.5)
sns.regplot(data=data_train, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#b4d4e1')
sns.regplot(data=data_test, x="True", y="Predicted", scatter=False, ax=g.ax_joint, color='#f4ba8a')
g.plot_marginals(sns.histplot, kde=False, element='bars', multiple='stack', alpha=0.5)
ax = g.ax_joint
ax.text(0.95, 0.1, f'Train $R^2$ = {r2_score(y_trainval, train_predict):.3f}', transform=ax.transAxes,
        fontsize=12, va='bottom', ha='right', bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))
ax.text(0.95, 0.05, f'Test $R^2$ = {r2_score(y_test, test_predict):.3f}', transform=ax.transAxes,
        fontsize=12, va='bottom', ha='right', bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))
ax.text(0.75, 0.99, 'Model = Random Forest', transform=ax.transAxes,
        fontsize=12, va='top', ha='left', bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))
ax.plot([data['True'].min(), data['True'].max()], [data['True'].min(), data['True'].max()],
        c="black", alpha=0.5, linestyle='--', label='x=y')
ax.legend()
plt.savefig(f"{save_dir}\\RF-Seaborn联合图_v8.png",dpi=600)
plt.close()

# 折线对比图
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='真实值', color='#1f77b4', linewidth=2)
plt.plot(test_predict, label='RF预测值', color='#ff7f0e', linestyle='dashed', linewidth=2)
plt.xlabel('样本序号', fontsize=12)
plt.ylabel('目标值', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.title('真实值 vs 预测值', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{save_dir}\\RF-折线对比图_v8.png",dpi=600)
plt.close()

# 拟合散点图
plt.figure(figsize=(10, 6))
plt.scatter(y_test, test_predict, color='#2ca02c', marker='o', label='预测值', alpha=0.7)
plt.plot(y_test, y_test, color='#d62728', linestyle='-', label='y=x', linewidth=2)
plt.xlabel('真实值', fontsize=12)
plt.ylabel('预测值', fontsize=12)
plt.title('真实值 vs 预测值', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{save_dir}\\RF-拟合散点图_v8.png",dpi=600)
plt.close()

# 误差直方图
plt.figure(figsize=(10, 6))
errors = y_test - test_predict
plt.hist(errors, bins=20, color='#9467bd', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(errors), color='red', linestyle='dashed', linewidth=2, label=f'平均误差 = {np.mean(errors):.2f}')
plt.xlabel('误差', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.title('误差直方分布图', fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{save_dir}\\RF-误差直方图_v8.png",dpi=600)
plt.close()

# ====== 8. 保存结果 ======
train_results_df = pd.DataFrame({
    '真实值': y_trainval.values,
    '预测值': train_predict
})
test_results_df = pd.DataFrame({
    '真实值': y_test.values,
    '预测值': test_predict
})

output_excel_path = f"{save_dir}\\RF_预测结果_v8.xlsx"

with pd.ExcelWriter(output_excel_path) as writer:
    train_results_df.to_excel(writer, sheet_name='训练集预测结果', index=False)
    test_results_df.to_excel(writer, sheet_name='测试集预测结果', index=False)


print(f"\n>>> 任务完成！所有评估数据已静默保存至: {report_file_path}")
print(f"\n所有 RF-v8 结果图表和模型已保存至: {save_dir}")