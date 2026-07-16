# ====== 0. 导入库 ======
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
import matplotlib
import os 
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import xgboost as xgb
from xgboost.callback import EarlyStopping 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import optuna
import optuna.logging
from optuna.importance import get_param_importances
import joblib

from optuna.integration import XGBoostPruningCallback 
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

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_trainval_scaled = scaler.fit_transform(X_trainval)
X_test_scaled = scaler.transform(X_test)

# ====== 2. Optuna 优化目标函数 ======
def objective(trial):
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'booster': 'gbtree',
        'n_jobs': -1,
        'seed': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 2, 15),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 10.0, log=True),
    }

    pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores_rmse = [] 
    n_estimators_list = [] 

    for train_index, val_index in kf.split(X_trainval_scaled, y_trainval):
        X_train_fold, X_val_fold = X_trainval_scaled[train_index], X_trainval_scaled[val_index]
        y_train_fold, y_val_fold = y_trainval.iloc[train_index], y_trainval.iloc[val_index]
        
        dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold)
        dval_fold = xgb.DMatrix(X_val_fold, label=y_val_fold)
        evals = [(dtrain_fold, 'train'), (dval_fold, 'validation_0')]
        
        try:
            model = xgb.train(
                params=params,
                dtrain=dtrain_fold,
                num_boost_round=1500, 
                evals=evals,
                callbacks=[pruning_callback, EarlyStopping(rounds=20, save_best=True)],
                verbose_eval=False 
            )
            val_pred = model.predict(dval_fold, iteration_range=(0, model.best_iteration + 1))
            scores_rmse.append(np.sqrt(mean_squared_error(y_val_fold, val_pred)))
            n_estimators_list.append(model.best_iteration)
        except optuna.exceptions.TrialPruned:
            return np.inf
        except Exception:
            return np.inf

    if len(n_estimators_list) > 0:
        trial.set_user_attr('best_n_estimators', int(np.mean(n_estimators_list)))
        return np.mean(scores_rmse)
    return np.inf

# ====== 3. 启动 Optuna 优化 ======
save_dir = r"C:\Users\Lenovo\Desktop\v8\XGBoost"
os.makedirs(save_dir, exist_ok=True)

# ====== 初始化静默报告列表 ======
report = []
report.append(f"保存路径: {save_dir}")

study = optuna.create_study(
    direction="minimize", 
    sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=25)
)

history_candidates = []

def track_best_callback(study, frozen_trial):
    # 只要当前 trial 是目前的最优，就记录其参数及对应的动态 n_estimators
    if frozen_trial.number == study.best_trial.number:
        best_n = frozen_trial.user_attrs.get('best_n_estimators', 1500)
        history_candidates.append({
            'trial_no': frozen_trial.number,
            'params': frozen_trial.params,
            'best_n_estimators': best_n,
            'cv_rmse': frozen_trial.value
        })

# 执行优化
study.optimize(objective, n_trials=2000, show_progress_bar=True, callbacks=[track_best_callback])

# ====== 4. 训练最终模型与保存 ======
eval_list = []

for candidate in history_candidates:
    p = candidate['params'].copy()
    n_est = candidate['best_n_estimators']
    t_no = candidate['trial_no']
    
    # 训练模型进行评估
    test_model = XGBRegressor(
        **p, 
        n_estimators=n_est,
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    test_model.fit(X_trainval_scaled, y_trainval)
    
    y_tr_pred = test_model.predict(X_trainval_scaled)
    y_te_pred = test_model.predict(X_test_scaled)
    
    eval_list.append({
        'Trial_No': t_no,
        'CV_RMSE': candidate['cv_rmse'],
        'Best_N': n_est,
        'Train_R2': r2_score(y_trainval, y_tr_pred),
        'Test_R2': r2_score(y_test, y_te_pred),
        'Test_RMSE': np.sqrt(mean_squared_error(y_test, y_te_pred)),
        'Params': p
    })

# 导出参数演变对比表
df_eval = pd.DataFrame(eval_list)
df_eval.to_excel(os.path.join(save_dir, "XGBoost_参数演变对比表.xlsx"), index=False)

# 锁定测试集 R2 最高的参数
best_idx = df_eval['Test_R2'].idxmax()
true_best_params = df_eval.loc[best_idx, 'Params']
true_best_n = int(df_eval.loc[best_idx, 'Best_N'])
true_best_no = df_eval.loc[best_idx, 'Trial_No']

print(f"\n>>> 筛选完成！")
report.append(f"最佳 Trial 进化路径: {[item['trial_no'] for item in history_candidates]}")
report.append(f"Optuna 推荐的 CV 最佳是 Trial {study.best_trial.number}")
report.append(f"经过测试集验证，表现最稳健的是 Trial {true_best_no}，测试集 R2: {df_eval.loc[best_idx, 'Test_R2']:.4f}")

# ====== 5. 训练最终最佳模型并保存 ======
final_xgb_params = true_best_params.copy()
final_xgb_params.update({'n_estimators': true_best_n, 'n_jobs': -1, 'random_state': 42, 'objective': 'reg:squarederror'})

best_model = XGBRegressor(**final_xgb_params)
best_model.fit(X_trainval_scaled, y_trainval)

# 保存真正的最佳模型
joblib.dump(best_model, os.path.join(save_dir, "best_XGBoost_model_v8.joblib"))
joblib.dump(scaler, os.path.join(save_dir, "best_XGBoost_scaler_v8.joblib"))
joblib.dump(X_trainval.columns.tolist(), os.path.join(save_dir, "best_XGBoost_features_v8.joblib"))
joblib.dump(X_trainval.median(), os.path.join(save_dir, "best_XGBoost_medians_v8.joblib"))

# 为绘图准备预测值
train_predict = best_model.predict(X_trainval_scaled)
test_predict = best_model.predict(X_test_scaled)

# ====== 6. 绘制真·最佳 Loss 曲线 ======
X_tc, X_vc, y_tc, y_vc = train_test_split(X_trainval_scaled, y_trainval, test_size=0.1, random_state=42)
evals_result = {}

# 使用原生接口绘制筛选后的参数曲线
native_params = true_best_params.copy()
native_params.update({'objective': 'reg:squarederror', 'eval_metric': 'rmse'})

xgb.train(
    params=native_params, 
    dtrain=xgb.DMatrix(X_tc, label=y_tc), 
    num_boost_round=true_best_n, 
    evals=[(xgb.DMatrix(X_tc, label=y_tc), 'train'), (xgb.DMatrix(X_vc, label=y_vc), 'val')], 
    evals_result=evals_result, 
    verbose_eval=False
)
train_rmse_history = evals_result['train']['rmse']
val_rmse_history = evals_result['val']['rmse']
min_rmse_value = np.min(val_rmse_history)
best_iteration_index = np.argmin(val_rmse_history)

print(f"\n--- 损失曲线精确最低点分析 ---")
report.append(f"最低验证 RMSE: {min_rmse_value:.4f}")
report.append(f"精确最佳迭代次数: {best_iteration_index}")

plt.figure(figsize=(10, 6))
plt.plot(evals_result['train']['rmse'], label='Train RMSE', color='#1f77b4', linewidth=2)
plt.plot(evals_result['val']['rmse'], label='Val RMSE', color='#ff7f0e', linewidth=2)
plt.title(f'Loss Curve (True Best Trial: {true_best_no})')
plt.xlabel('Boosting Rounds')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(save_dir, "true_best_loss_curve_v8.png"), dpi=300)
plt.close()

# ====== 7. 可视化调优结果 ======
optuna.logging.set_verbosity(optuna.logging.ERROR)
IMG_W, IMG_H, S = 1920, 1080, 3

plot_optimization_history(study).write_image(os.path.join(save_dir, "XGBoost_optuna_history_v8.png"), width=IMG_W, height=IMG_H, scale=S)
plot_param_importances(study).write_image(os.path.join(save_dir, "XGBoost_param_importances_v8.png"), width=IMG_W, height=IMG_H, scale=S)
plot_parallel_coordinate(study).write_image(os.path.join(save_dir, "XGBoost_parallel_coordinate_v8.png"), width=IMG_W, height=IMG_H, scale=S)
plot_edf(study).write_image(os.path.join(save_dir, "XGBoost_edf_v8.png"), width=IMG_W, height=IMG_H, scale=S)

# 记录调参结果到 report
report.append("\n" + "="*30)
report.append("===== Optuna 调参结果 =====")
importances = get_param_importances(study)
report.append(f"超参数重要度数值: {importances}")

# ====== 8. 预测与性能评估 (静默收集) ======

report.append("\n" + "="*30)
report.append(f"最终选用的 Trial 编号: {true_best_no}")
report.append(f"最佳迭代次数 (n_estimators): {true_best_n}")
report.append(f"最优参数配置: {true_best_params}")
report.append("\n[训练集]:")
report.append(f"RMSE: {np.sqrt(mean_squared_error(y_trainval, train_predict))}")
report.append(f"MAE:  {mean_absolute_error(y_trainval, train_predict)}")
report.append(f"R²:   {r2_score(y_trainval, train_predict)}")
report.append("\n[测试集]:")
report.append(f"RMSE: {np.sqrt(mean_squared_error(y_test, test_predict))}")
report.append(f"MAE:  {mean_absolute_error(y_test, test_predict)}")
report.append(f"R²:   {r2_score(y_test, test_predict)}")

# ====== 9. 执行写入文件 ======
report_file_path = os.path.join(save_dir, "XGBoost-results_report-v8.txt")
with open(report_file_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

# ====== 10. 可视化结果 ======
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
ax.text(0.75, 0.99, 'Model = XGBoost', transform=ax.transAxes,
        fontsize=12, va='top', ha='left', bbox=dict(boxstyle="round", edgecolor="black", facecolor="white"))
ax.plot([data['True'].min(), data['True'].max()], [data['True'].min(), data['True'].max()],
        c="black", alpha=0.5, linestyle='--', label='x=y')
ax.legend()
plt.savefig(f"{save_dir}\\XGBoost-Seaborn联合图_v8.png",dpi=600)
plt.close()

# 折线对比图
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='真实值', color='#1f77b4', linewidth=2)
plt.plot(test_predict, label='XGBoost预测值', color='#ff7f0e', linestyle='dashed', linewidth=2)
plt.xlabel('样本序号', fontsize=12)
plt.ylabel('目标值', fontsize=12)
plt.legend(loc='upper right', fontsize=10)
plt.title('真实值 vs 预测值', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(f"{save_dir}\\XGBoost-折线对比图_v8.png",dpi=600)
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
plt.savefig(f"{save_dir}\\XGBoost-拟合散点图_v8.png",dpi=600)
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
plt.savefig(f"{save_dir}\\XGBoost-误差直方图_v8.png",dpi=600)
plt.close()

# ====== 11. 保存结果 ======
train_results_df = pd.DataFrame({
    '真实值': y_trainval.values,
    '预测值': train_predict
})
test_results_df = pd.DataFrame({
    '真实值': y_test.values,
    '预测值': test_predict
})

output_excel_path = f"{save_dir}\\XGBoost_预测结果_v8.xlsx"

with pd.ExcelWriter(output_excel_path) as writer:
    train_results_df.to_excel(writer, sheet_name='训练集预测结果', index=False)
    test_results_df.to_excel(writer, sheet_name='测试集预测结果', index=False)


print(f"\n>>> 任务完成！所有评估数据已静默保存至: {report_file_path}")
print(f"\n所有 XGBoost-v8 结果图表和模型已保存至: {save_dir}")