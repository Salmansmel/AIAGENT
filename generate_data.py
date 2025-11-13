import pandas as pd
import numpy as np

# تعريف الأعمدة والخيارات
job_roles = ['مطور', 'مهندس أول', 'مدير مشروع', 'محلل بيانات', 'مصمم', 'مهندس برمجيات']
stress_levels = ['مسيطر', 'عادي', 'مضغوط']
employee_ids = [f'E{1001 + i}' for i in range(1000)]

def generate_employee_data(num_rows=1000):
    data = []
    
    for i in range(num_rows):
        role = np.random.choice(job_roles)
        fatigue = np.random.uniform(1.0, 10.0)
        satisfaction = np.random.randint(1, 6)
        after_hours = np.random.randint(0, 15)
        task_count = np.random.randint(3, 15)
        collaboration = np.random.randint(5, 30)
        tenure = np.random.uniform(0.5, 10.0)

        # 2. إنشاء "منطق" للاحتراق
        burnout_score = 0
        if fatigue > 6.5: burnout_score += 3
        if after_hours > 8: burnout_score += 3
        if satisfaction < 2: burnout_score += 2
        if collaboration > 22: burnout_score += 1
        if task_count > 12: burnout_score += 1

        # 3. جعل البيانات "أكثر فوضوية" (للحصول على دقة واقعية)
        if np.random.rand() > 0.3: # 70% فرصة
             # التغيير هنا: زدنا نطاق العشوائية لجعل البيانات "أصعب"
             burnout_score += np.random.randint(-3, 3)
        
        burnout_risk = 1 if burnout_score >= 5 else 0
        stress = 'مضغوط' if fatigue > 6.0 else np.random.choice(['مسيطر', 'عادي'])
        
        data.append({
            'employee_id': f'E{1001 + i}',
            'weekly_task_count': task_count,
            'avg_task_complexity': np.random.randint(2, 6),
            'collaboration_load': collaboration,
            'after_hours_work': after_hours,
            'resource_allocation': np.random.randint(30, 51),
            'tenure_in_role': round(tenure, 1),
            'job_role': role,
            'daily_stress_pulse': stress,
            'mental_fatigue_score': round(fatigue, 1),
            'job_satisfaction_rating': satisfaction,
            'burnout_risk': burnout_risk
        })

    return pd.DataFrame(data)

# --- التشغيل ---
if __name__ == "__main__":
    print("... جارٍ إنشاء 1000 صف بيانات (أكثر فوضوية)...")
    large_dataset = generate_employee_data(1000)
    
    file_name = 'employee_data_large.csv'
    large_dataset.to_csv(file_name, index=False, encoding='utf-8-sig')
    
    print(f"✅ تم بنجاح!")
    print(f"تم إنشاء ملف '{file_name}' ببيانات فوضوية وواقعية.")
    print("\n--- توزيع البيانات (لمعرفة التوازن) ---")
    print(large_dataset['burnout_risk'].value_counts(normalize=True))