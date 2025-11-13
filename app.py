import streamlit as st
import pandas as pd
import numpy as np
import io

from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# (دالة تدريب النموذج لا تتغير)
def train_burnout_model(df):
    numeric_features = [
        'weekly_task_count', 'avg_task_complexity', 'collaboration_load',
        'after_hours_work', 'resource_allocation', 'tenure_in_role',
        'mental_fatigue_score', 'job_satisfaction_rating'
    ]
    categorical_features = ['job_role', 'daily_stress_pulse']
    target = 'burnout_risk'
    X = df[numeric_features + categorical_features]
    y = df[target]
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    param_grid = {
        'model__n_estimators': [50, 100],
        'model__max_depth': [10, None],
        'model__min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(
        estimator=pipeline, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1
    )
    grid_search.fit(X, y)
    best_model = grid_search.best_estimator_
    best_score = grid_search.best_score_
    st.write(f"أفضل إعدادات للنموذج (القوي): {grid_search.best_params_}")
    return best_model, best_score

# (دالة التنبؤ لا تتغير)
def get_burnout_prediction(model_pipeline, employee_input_data):
    input_df = pd.DataFrame([employee_input_data])
    probabilities = model_pipeline.predict_proba(input_df)
    return probabilities[0][1]

# (دالة المؤشر لا تتغير)
def get_risk_indicator(risk_score):
    if risk_score < 0.4: return "🟢 مستقر"
    elif risk_score < 0.7: return "🟡 متوسط"
    else: return "🔴 مرتفع"

# (فئة الوكيل لا تتغير)
class BurnoutShieldAgent:
    def __init__(self, employee_data, model):
        self.employees = employee_data
        self.model = model

    def _calculate_suitability_score(self, employee, task_skills, task_hours, task_importance):
        emp_skills = employee.get('skills', [])
        if not isinstance(emp_skills, list): emp_skills = []
        skill_match = 1.0 if task_skills in emp_skills else 0.1
        current_risk = employee['burnout_risk_prob']
        future_workload = employee['weekly_task_count'] + 1
        avg_hours = employee['avg_hours_per_task']
        is_low_impact_task = (task_hours <= (avg_hours * 0.5))
        risk_penalty = current_risk * 20
        if current_risk > 0.6 and is_low_impact_task and task_importance != "عالية":
            risk_penalty = risk_penalty * 0.3
        if employee['state'] == 'focus_mode': return -999
        if employee['state'] == 'request_variety' and skill_match < 1.0: skill_match = 1.2
        score = (skill_match * 10) - risk_penalty - (future_workload / 15.0 * 5)
        return score

    def suggest_employees(self, task_skills, task_hours, task_importance):
        suggestions = []
        for index, employee in self.employees.iterrows():
            score = self._calculate_suitability_score(employee, task_skills, task_hours, task_importance)
            if score > -999:
                suggestions.append((employee['name'], score, employee['burnout_risk_indicator']))
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions[:3]

    def check_assignment_warning(self, employee_id, task_skills, task_hours, task_importance):
        """
        (محدث بمعادلة ديناميكية)
        يحسب تأثير المهمة كدالة للأهمية والساعات.
        """
        employee = self.employees.loc[employee_id]
        current_risk_prob = employee['burnout_risk_prob']

        # --- (1) التغيير هنا: إنشاء "سيناريو مستقبلي" بالمعادلة ---
        hypothetical_data = employee.to_dict()
        
        # 1. تحديثات المهام (كما كانت)
        hypothetical_data['weekly_task_count'] += 1
        hypothetical_data['after_hours_work'] += (task_hours * 0.25)
        hypothetical_data['collaboration_load'] += 3

        # 2. (المعادلة الجديدة) محاكاة الإرهاق كدالة للأهمية والساعات
        
        # 2a. تحديد "وزن" الأهمية
        if task_importance == "عالية":
            importance_weight = 3.0  # تأثير أساسي كبير
        elif task_importance == "متوسطة":
            importance_weight = 1.5  # تأثير أساسي متوسط
        else: # منخفضة
            importance_weight = 0.5  # تأثير أساسي منخفض
            
        # 2b. تحديد "عامل الساعات" (بافتراض أن 8 ساعات هي 100% من التأثير)
        hour_factor = task_hours / 8.0 
        
        # 2c. المعادلة: الإرهاق المضاف = الوزن * عامل الساعات
        added_fatigue = importance_weight * hour_factor
        
        current_fatigue = employee['mental_fatigue_score']
        hypothetical_data['mental_fatigue_score'] = min(current_fatigue + added_fatigue, 10.0)
            
        # 3. (المنطق الجديد) محاكاة "الرضا الوظيفي" (يبقى كما هو)
        current_satisfaction = employee['job_satisfaction_rating']
        if task_importance == "عالية" and current_fatigue < 7:
             hypothetical_data['job_satisfaction_rating'] = min(current_satisfaction + 0.5, 5.0)
        elif task_importance == "منخفضة" and current_fatigue > 7:
             hypothetical_data['job_satisfaction_rating'] = max(current_satisfaction - 0.5, 1.0)


        # --- (2) التنبؤ (رأي النموذج) ---
        predicted_new_risk_prob = get_burnout_prediction(self.model, hypothetical_data)

        # --- (3) تطبيق "طبقة الحماية المنطقية" (تبقى كما هي) ---
        final_new_risk_prob = predicted_new_risk_prob
        
        if current_risk_prob >= 0.70 and predicted_new_risk_prob < current_risk_prob:
            final_new_risk_prob = min(current_risk_prob + (added_fatigue * 0.05), 1.0) # <-- ربط العقوبة بالإرهاق المضاف
            
            print(f"** تنبيه منطقي **: تم اكتشاف حالة شاذة للموظف {employee_id}.")
            print(f"   الخطر الحالي: {current_risk_prob:.1%}, تنبؤ النموذج: {predicted_new_risk_prob:.1%}")
            print(f"   تم التجاوز إلى: {final_new_risk_prob:.1%}")

        # --- (4) التحقق من الاستثناء (مهمة صغيرة جداً) (يبقى كما هو) ---
        if current_risk_prob >= 0.7:
            avg_hours = employee['avg_hours_per_task']
            is_low_impact_task = (task_hours <= (avg_hours * 0.5))
            if is_low_impact_task:
                if task_importance == "منخفضة":
                    return f"🟡 **تحذير (خطر مُدار):** {employee['name']} في **خطر مرتفع** ({current_risk_prob:.0%}).\n\n**ولكن:** هذه المهمة ({task_hours} س) صغيرة جداً مقارنة بمتوسطه ({avg_hours:.1f} س/مهمة) و**منخفضة الأهمية**.\n\n**النتيجة:** إسنادها **ممكن**، والخطر المستقبلي المتوقع سيظل **{final_new_risk_prob:.0%}**."
                if task_importance == "متوسطة":
                    return f"🟠 **تحذير (يُفضل تجنب):** {employee['name']} في **خطر مرتفع** ({current_risk_prob:.0%}).\n\nالمهمة صغيرة ({task_hours} س)، لكنها **متوسطة الأهمية**. الخطر المستقبلي المتوقع **{final_new_risk_prob:.0%}**."

        # --- (5) إصدار التحذير الرئيسي (يبقى كما هو) ---
        if final_new_risk_prob > current_risk_prob:
            if final_new_risk_prob >= 0.7 and current_risk_prob < 0.7:
                 return f"⚠️ **تحذير استباقي:** {employee['name']} **آمن حالياً** ({current_risk_prob:.0%}).\n\n**ولكن:** إسناد هذه المهمة سينقله إلى منطقة **الخطر المرتفع** (المتوقع: **{final_new_risk_prob:.0%}**)."
            else:
                 return f"🚨 **خطر مرتفع (لا يُنصح به):** {employee['name']} يعاني من ضغط حالياً ({current_risk_prob:.0%}).\n\nإضافة هذه المهمة سترفع الخطر إلى **{final_new_risk_prob:.0%}**."
        else:
            return f"✅ **آمن:** {employee['name']} في وضع {get_risk_indicator(current_risk_prob)} ({current_risk_prob:.0%}).\n\nإسناد المهمة يحافظ عليه في وضع آمن (الخطر المتوقع: **{final_new_risk_prob:.0%}**)."
def load_raw_data():
    file_name = 'employee_data_large.csv'
    try:
        df = pd.read_csv(file_name)
        return df
    except FileNotFoundError:
        st.error(f"خطأ: لم يتم العثور على ملف '{file_name}'. يرجى إنشاء الملف أولاً (بتشغيل generate_data.py).")
        return None

# (الدالة الرئيسية)
def main():
    st.set_page_config(layout="wide", page_title="درع الاحتراق الوظيفي")
    
    raw_data = load_raw_data()
    if raw_data is None: return

    if 'model' not in st.session_state:
        # --- (1) التغيير هنا: سيعاد التدريب على البيانات "الفوضوية" ---
        with st.spinner("⏳ جارٍ تدريب النموذج القوي على 1000 صف (بيانات واقعية فوضوية)..."):
            model, accuracy = train_burnout_model(raw_data)
            st.session_state.model = model
            st.session_state.model_accuracy = accuracy
    
    st.sidebar.header("محاكاة أدوات الموظف")
    st.sidebar.info("هذا الجزء يحاكي إدخالات الموظفين من تطبيقهم الخاص.")
    # --- (2) التغيير هنا: ستظهر دقة "واقعية" (أقل) ---
    st.sidebar.success(f"دقة النموذج (الواقعية): {st.session_state.model_accuracy:.1%}")
    
    if 'employee_data' not in st.session_state:
        all_skills = ['Python', 'SQL', 'UI/UX', 'Figma', 'Cloud', 'Analysis', 'Security', 'DevOps']
        generated_skills = [
            np.random.choice(all_skills, np.random.randint(1, 3), replace=False).tolist()
            for _ in range(len(raw_data))
        ]
        raw_data['skills'] = generated_skills
        raw_data['avg_hours_per_task'] = raw_data['resource_allocation'] / (raw_data['weekly_task_count'] + 0.01)
        raw_data['name'] = raw_data['employee_id'] 
        raw_data['state'] = 'available' 
        
        with st.spinner("⏳ جارٍ حساب مؤشر الخطر الأولي لجميع الموظفين..."):
            raw_data['burnout_risk_prob'] = raw_data.apply(
                lambda row: get_burnout_prediction(st.session_state.model, row.to_dict()),
                axis=1
            )
        
        raw_data['burnout_risk_indicator'] = raw_data['burnout_risk_prob'].apply(get_risk_indicator)
        st.session_state.employee_data = raw_data

    
    employee_list = st.session_state.employee_data['name'].tolist()
    selected_employee_name = st.sidebar.selectbox(
        "اختر الموظف لتحديث حالته (عينة 50 موظف):", 
        employee_list[:50]
    )
    selected_employee_id = st.session_state.employee_data[st.session_state.employee_data['name'] == selected_employee_name].index[0]
    
    # (بقية كود الشريط الجانبي لا يتغير)
    fatigue_map = {"🙂 مسيطر": 2.0, "😐 عادي": 4.0, "😥 مضغوط": 7.0, "😫 مضغوط جداً": 9.0}
    current_fatigue_val = st.session_state.employee_data.loc[selected_employee_id, 'mental_fatigue_score']
    current_fatigue_desc = min(fatigue_map.keys(), key=lambda k: abs(fatigue_map[k] - current_fatigue_val))
    new_fatigue_desc = st.sidebar.radio(f"تحديث الحالة النفسية لـ {selected_employee_name}:", options=fatigue_map.keys(), index=list(fatigue_map.keys()).index(current_fatigue_desc))
    new_fatigue_val = fatigue_map[new_fatigue_desc]
    new_stress_pulse = "مضغوط" if new_fatigue_val >= 6.0 else "مسيطر"
    state_map = {"🟢 متاح": 'available', "🔵 وضع التركيز": 'focus_mode', "🎨 طلب تنوع": 'request_variety'}
    current_state_desc = [k for k, v in state_map.items() if v == st.session_state.employee_data.loc[selected_employee_id, 'state']][0]
    new_state_desc = st.sidebar.selectbox(f"تحديث وضع العمل لـ {selected_employee_name}:", options=state_map.keys(), index=list(state_map.keys()).index(current_state_desc))
    new_state_val = state_map[new_state_desc]
    if st.sidebar.button("تحديث بيانات الموظف"):
        st.session_state.employee_data.loc[selected_employee_id, 'mental_fatigue_score'] = new_fatigue_val
        st.session_state.employee_data.loc[selected_employee_id, 'daily_stress_pulse'] = new_stress_pulse
        st.session_state.employee_data.loc[selected_employee_id, 'state'] = new_state_val
        updated_data_row = st.session_state.employee_data.loc[selected_employee_id].to_dict()
        new_risk_score = get_burnout_prediction(st.session_state.model, updated_data_row)
        st.session_state.employee_data.loc[selected_employee_id, 'burnout_risk_prob'] = new_risk_score
        st.session_state.employee_data.loc[selected_employee_id, 'burnout_risk_indicator'] = get_risk_indicator(new_risk_score)
        st.rerun()
    
    st.title("🛡️ لوحة تحكم درع الاحتراق الوظيفي (نموذج واقعي)")
    
    st.header("🩺 لوحة صحة الفريق")
    display_columns = [
        'name', 'burnout_risk_indicator', 'burnout_risk_prob', 'weekly_task_count', 
        'avg_hours_per_task', 'state', 'skills'
    ]
    st.subheader("عرض عينة (أول 100 موظف) من 1000")
    styled_df = st.session_state.employee_data.head(100)[display_columns].style.format({
        'burnout_risk_prob': '{:.1%}', 'avg_hours_per_task': '{:.1f} س/مهمة'
    })
    st.dataframe(styled_df, use_container_width=True)

    agent = BurnoutShieldAgent(st.session_state.employee_data, st.session_state.model)
    
    # --- (3) قسم "المستشار" (لا تغيير كبير) ---
    st.header("🧠 الـ AI المستشار: اقتراح مهمة جديدة")
    all_skills_list = ['Python', 'SQL', 'UI/UX', 'Figma', 'Cloud', 'Analysis', 'Security', 'DevOps']
    with st.form("task_suggestion_form"):
        st.write("أدخل تفاصيل المهمة الجديدة والوكيل سيقترح الموظف الأنسب صحياً.")
        col1, col2, col3 = st.columns(3)
        task_skill = col1.selectbox("المهارة المطلوبة:", all_skills_list)
        task_hours = col2.slider("الساعات المقدرة للمهمة:", 1, 10, 2)
        task_importance = col3.selectbox("أهمية المهمة:", ["منخفضة", "متوسطة", "عالية"], index=0)
        submitted = st.form_submit_button("💡 الحصول على اقتراح")
        if submitted:
            with st.spinner("🧠 الوكيل يبحث في 1000 موظف عن الأنسب..."):
                suggestions = agent.suggest_employees(task_skill, task_hours, task_importance)
            st.subheader("أفضل الاقتراحات (الأعلى صحة وملاءمة):")
            if suggestions:
                for name, score, risk in suggestions:
                    st.success(f"**{name}** - (مؤشر الخطر: {risk})")
            else:
                st.warning("لا يوجد موظفون متاحاً حالياً يطابقون هذه المعير.")

    # --- (4) التغيير هنا: تعديل قسم "التحقق الاستباقي" ---
    st.header("❗ التحقق الاستباقي (الإسناد اليدوي)")
    st.write("اختر موظفاً أولاً، ثم ستظهر مهاراته لإجراء الفحص.")
    
    # 1. اختيار الموظف (خارج الفورم)
    manual_employee_name = st.selectbox("1. اختر الموظف (عينة 50):", employee_list[:50])
    
    # 2. جلب بيانات هذا الموظف
    employee_details = st.session_state.employee_data[
        st.session_state.employee_data['name'] == manual_employee_name
    ].iloc[0]
    
    # 3. جلب مهاراته الخاصة
    employee_skills = employee_details.get('skills', [])
    if not employee_skills:
        employee_skills = ["لا توجد مهارات مسجلة"]

    # 4. الآن يبدأ الفورم
    with st.form("manual_assignment_form"):
        st.write(f"**2. اختر مهمة من مهارات {manual_employee_name}**:")
        col1, col2, col3 = st.columns(3)
        
        # --- (5) التغيير هنا: القائمة المنسدلة تعرض مهارات الموظف فقط ---
        manual_task_skill = col1.selectbox("المهارة للمهمة (من مهاراته):", employee_skills)
        manual_task_hours = col2.slider("الساعات المقدرة:", 1, 10, 2)
        manual_task_importance = col3.selectbox("أهمية المهمة:", ["منخفضة", "متوسطة", "عالية"], index=0)
        
        check_submitted = st.form_submit_button("🔍 فحص إمكانية الإسناد")
        
        if check_submitted:
            manual_employee_id = employee_details.name # (نستخدم الاسم الحقيقي ID)
            
            warning = agent.check_assignment_warning(
                manual_employee_id, manual_task_skill, manual_task_hours, manual_task_importance
            )
            # عرض التحذير
            if "✅ آمن" in warning: st.success(warning)
            elif "🟡 تحذير (خطر مُدار)" in warning: st.warning(warning)
            elif "🟠 تحذير (يُفضل تجنب)" in warning: st.warning(warning)
            elif "⚠️ تحذير استباقي" in warning: st.warning(warning)
            elif "🚨 خطر مرتفع" in warning: st.error(warning)
            else: st.info(warning)

if __name__ == "__main__":
    main()