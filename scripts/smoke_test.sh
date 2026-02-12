#!/bin/bash
# =============================================
# smoke_test.sh — Quick validation of the model API
# =============================================
# Prerequisite: model-server must be running on port 5001
# Usage: bash scripts/smoke_test.sh

echo ""
echo "=== Smoke Test: Churn Prediction API ==="
echo ""
echo "Testing endpoint: http://localhost:5001/invocations"
echo ""

RESPONSE=$(curl -s -X POST http://localhost:5001/invocations \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges", "gender_Male", "Partner_Yes", "Dependents_Yes", "PhoneService_Yes", "MultipleLines_No phone service", "MultipleLines_Yes", "InternetService_Fiber optic", "InternetService_No", "OnlineSecurity_No internet service", "OnlineSecurity_Yes", "OnlineBackup_No internet service", "OnlineBackup_Yes", "DeviceProtection_No internet service", "DeviceProtection_Yes", "TechSupport_No internet service", "TechSupport_Yes", "StreamingTV_No internet service", "StreamingTV_Yes", "StreamingMovies_No internet service", "StreamingMovies_Yes", "Contract_One year", "Contract_Two year", "PaperlessBilling_Yes", "PaymentMethod_Credit card (automatic)", "PaymentMethod_Electronic check", "PaymentMethod_Mailed check"], "data": [[0, 2, 70.0, 140.0, true, false, false, true, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, true, false]]}}')

echo "Response: $RESPONSE"

if echo "$RESPONSE" | grep -q "predictions"; then
  echo ""
  echo "✅ Smoke test PASSED"
else
  echo ""
  echo "❌ Smoke test FAILED"
  exit 1
fi
