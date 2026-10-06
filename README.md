# Phân tích rủi ro tín dụng: từ EDA đến Scorecard

Dự án tự học Data Analyst trên bộ dữ liệu Kaggle *credit-risk-classification* (`customer_data.csv`, `payment_data.csv`): 1.125 khách hàng, 20% thuộc nhóm rủi ro cao.

## Nội dung notebook
1. **EDA:** so sánh nhóm rủi ro cao/thấp theo quá hạn và dư nợ, tương quan Spearman, xử lý dữ liệu thiếu.
2. **Feature engineering:** gộp 8.250 dòng `payment_data` thành 25 đặc trưng mức khách hàng (tổng quá hạn, tỷ lệ dư nợ/hạn mức, số sản phẩm, dư nợ cao nhất, tỷ lệ trả đúng hạn…).
3. **Mô hình:** Logistic Regression, Random Forest, LightGBM, XGBoost; xử lý mất cân bằng bằng `class_weight` và SMOTE; đánh giá bằng ROC-AUC, PR-AUC, Recall, Gini, KS với cross-validation 5 fold.
4. **Giải thích mô hình:** SHAP (toàn cục và từng khách hàng).
5. **Scorecard WoE/IV:** bảng điểm (PDO 20, 600 điểm = odds 50:1) và chọn ngưỡng duyệt vay theo chi phí.

## Kết quả chính
| | Kết quả |
|---|---|
| Mô hình tốt nhất theo CV | Random Forest, AUC 0.69, Gini 0.37 |
| Scorecard 9 biến (test) | AUC 0.72, Gini 0.45 |
| Ngưỡng duyệt (chi phí FN:FP = 5:1) | 535 điểm: duyệt ~47% hồ sơ, tỷ lệ xấu trong nhóm duyệt giảm từ 20% xuống 11%, chi phí giảm 34% |
| Yếu tố rủi ro chính (IV và SHAP) | tỷ lệ dư nợ/hạn mức cao, `fea_4` thấp, ít lần trả đúng hạn |

## Cấu trúc
```
credit_risk_analysis.ipynb   notebook chính (đã chạy sẵn kết quả)
credit_risk_analysis.html    bản xem nhanh trên trình duyệt
data/                        dữ liệu gốc
outputs/                     biểu đồ, bảng IV, scorecard, kết quả mô hình
```

## Chạy lại
```bash
pip install -r requirements.txt
jupyter notebook credit_risk_analysis.ipynb
```
