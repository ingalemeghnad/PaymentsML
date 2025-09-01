import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
faker = Faker()
Faker.seed(42)
np.random.seed(42)
random.seed(42)

N_DEBTORS = 500
N_TXNS = 50000
ANOMALY_RATE = 0.02
channels = ["mobile","internet","batch","internal"]
currencies = ["GBP","EUR","USD","JPY"]
debtors = [f"D{i:04d}" for i in range(1, N_DEBTORS+1)]
records = []
start = datetime(2024,1,1)
for i in range(N_TXNS):
    debtor = random.choice(debtors)
    seg = random.choices(["RETAIL","SME","CORP"],[0.7,0.25,0.05])[0]
    if seg=="RETAIL":
        amount = max(5, np.random.normal(300, 150))
    elif seg=="SME":
        amount = max(20, np.random.normal(3000, 1500))
    else:
        amount = max(1000, np.random.normal(40000, 20000))
    channel = random.choice(channels)
    currency = random.choice(currencies[:3])
    creditor_name = faker.company()
    creditor_account = faker.iban()
    remittance_info = random.choice([
        f"Invoice {random.randint(1000,9999)}",
        f"Salary {random.randint(1,12)}-{random.randint(2023,2025)}",
        "Payment",
        "Refund"
    ])
    ts = start + timedelta(minutes=random.randint(0,60*24*180))
    execution_time = ts.strftime('%H:%M')
    country = random.choice(["UK","DE","FR","US"])
    is_anom = 0
    reason = ""

    # inject anomalies with probability
    if random.random() < ANOMALY_RATE:
        is_anom = 1
        t = random.choice(["high_amt","currency","odd_hours","new_creditor","weird_remit","time_exec","burst"])
        if t=="high_amt":
            if seg=="RETAIL":
                amount = random.randint(50000,200000)
            elif seg=="SME":
                amount = random.randint(200000,800000)
            else:
                amount = random.randint(1000000,5000000)
            reason = "Unusually high payment"
        elif t=="currency":
            currency = "JPY"
            reason = "Unusual currency for debtor"
        elif t=="odd_hours":
            ts = ts.replace(hour=random.randint(1,3), minute=random.randint(0,59))
            execution_time = ts.strftime('%H:%M')
            reason = "Burst of payments at odd hours"
        elif t=="new_creditor":
            creditor_name = faker.company() + " NEW"
            creditor_account = faker.iban()
            reason = "Unusual creditor"
        elif t=="weird_remit":
            remittance_info = faker.lexify(text='??????')
            reason = "Odd remittance pattern"
        elif t=="time_exec":
            execution_time = f"{random.randint(0,23):02d}:{random.randint(0,59):02d}"
            reason = "Time of execution anomaly"
        elif t=="burst":
            ts = ts.replace(hour=2, minute=random.randint(0,4))
            execution_time = ts.strftime('%H:%M')
            reason = "Burst of payments (micro-burst)"

    records.append({
        'txn_id': f"T{1000000+i}",
        'debtor_id': debtor,
        'segment': seg,
        'amount': round(float(amount),2),
        'currency': currency,
        'channel': channel,
        'creditor_name': creditor_name,
        'creditor_account': creditor_account,
        'remittance_info': remittance_info,
        'execution_time': execution_time,
        'timestamp': ts.isoformat(),
        'country': country,
        'is_anomaly': is_anom,
        'reason': reason
    })
# Write CSV
import os
out = 'synthetic_payments_enriched.csv'
df = pd.DataFrame.from_records(records)
df.to_csv(out, index=False)
print(f"Wrote {len(df)} rows to {out}")
