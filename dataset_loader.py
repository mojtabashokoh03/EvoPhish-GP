# dataset_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_prepare_dataset(dataset_name, file_path=None, target_column="label"):
    """
    بارگذاری و پیش‌پردازش دیتاست‌های مختلف برای کد GP
    
    Parameters:
    -----------
    dataset_name : str
        نام دیتاست ('PWD2021', 'phiushiil', 'iscx')
    file_path : str, optional
        مسیر فایل (اگر None باشد از مسیر پیش‌فرض استفاده می‌کند)
    target_column : str
        نام ستون هدف
    
    Returns:
    --------
    X : pd.DataFrame
        ویژگی‌ها
    y : pd.Series
        برچسب‌ها (0=عادی, 1=فیشینگ) ← این استاندارد است
    used_columns : list
        لیست ستون‌های استفاده شده
    """
    
    # مسیرهای پیش‌فرض
    default_paths = {
        'PWD2021': "PWD2021.csv",
        'phiushiil': "PhiUSIIL_Phishing_URL_Dataset.csv",
        'iscx': "AllISCX.csv"
    }
    
    if file_path is None:
        file_path = default_paths.get(dataset_name.lower())
        if file_path is None:
            raise ValueError(f"دیتاست {dataset_name} شناخته شده نیست. گزینه‌ها: PWD2021, phiushiil, iscx")
    
    print(f"\n[بارگذاری] دیتاست: {dataset_name}")
    print(f"[مسیر] {file_path}")
    
    # =========================================================
    # دیتاست اول: PWD2021 (ساختار اصلی کد)
    # =========================================================
    if dataset_name.lower() == 'PWD2021':
        df = pd.read_csv(file_path)
        
        # بررسی وجود ستون هدف
        if target_column not in df.columns:
            possible_targets = ['isphish', 'phishing', 'label', 'target', 'class']
            for col in possible_targets:
                if col in df.columns:
                    target_column = col
                    break
            else:
                raise KeyError(f"ستون هدف در دیتاست PWD2021 یافت نشد. ستون‌ها: {df.columns.tolist()}")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # کد کردن ویژگی‌های categorical
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        
        X = X.fillna(0)
        
        # اطمینان از استاندارد بودن برچسب‌ها: 1=phishing, 0=benign
        unique_vals = y.unique()
        print(f"[برچسب‌های اصلی] {unique_vals}")
        
        # اگر برچسب‌ها معکوس بودند، اصلاح کن
        if set(unique_vals) == {0, 1}:
            # بررسی میانگین: اگر میانگین > 0.5 یعنی بیشتر نمونه‌ها phishing هستند
            # اما در دیتاست‌های فیشینگ معمولاً تعادل وجود دارد، پس بررسی می‌کنیم
            phishing_ratio = y.mean()
            if phishing_ratio < 0.3:
                # احتمالاً برچسب‌ها معکوس هستند (چون انتظار داریم فیشینگ زیاد باشد)
                print(f"[هشدار] نسبت فیشینگ بسیار کم است ({phishing_ratio:.2f}) - معکوس کردن برچسب‌ها")
                y = 1 - y
        
        print(f"[اطلاعات] نمونه‌ها: {len(X)}, ویژگی‌ها: {X.shape[1]}")
        print(f"[اطلاعات] توزیع نهایی: عادی={sum(y==0)}, فیشینگ={sum(y==1)}")
        
        return X, y, X.columns.tolist()
    
    # =========================================================
    # دیتاست دوم: PhiUSIIL_Phishing_URL_Dataset
    # =========================================================
    elif dataset_name.lower() == 'phiushiil':
        df = pd.read_csv(file_path)
        
        print(f"[اطلاعات] شکل دیتاست: {df.shape}")
        
        # ستون هدف 'label' است
        target_col = 'label'
        
        if target_col not in df.columns:
            raise KeyError(f"ستون '{target_col}' در دیتاست PhiUSIIL یافت نشد")
        
        y_raw = df[target_col].copy()
        
        # بررسی برچسب‌های اصلی
        print(f"[برچسب‌های اصلی] {y_raw.unique()}")
        print(f"[توزیع اصلی] 0={sum(y_raw==0)}, 1={sum(y_raw==1)}")
        
        # در دیتاست PhiUSIIL: 0 = phishing, 1 = benign (معکوس است!)
        # ما نیاز داریم: 1 = phishing, 0 = benign
        # بنابراین باید معکوس کنیم
        y = 1 - y_raw  # معکوس کردن: 0->1 (phishing), 1->0 (benign)
        
        print(f"[برچسب‌های پس از اصلاح] 0=benign, 1=phishing")
        print(f"[توزیع اصلاح شده] benign={sum(y==0)}, phishing={sum(y==1)}")
        
        # حذف ستون‌های غیرعددی و نویز
        columns_to_drop = [target_col]
        
        # حذف ستون Title چون حاوی متن فارسی/تایلندی است
        if 'Title' in df.columns:
            columns_to_drop.append('Title')
            print("[حذف] ستون Title (حاوی متن نویز)")
        
        # حذف ستون URL اگر باشد
        text_columns = ['URL', 'Domain', 'url', 'domain','FILENAME','TLD','URLSimilarityIndex','TLDLegitimateProb','URLCharProb','DomainTitleMatchScore','URLTitleMatchScore','IsHTTPS','Title']
        for col in text_columns:
            if col in df.columns:
                columns_to_drop.append(col)
                print(f"[حذف] ستون {col}")
        
        X = df.drop(columns=columns_to_drop)
        
        # اطمینان از عددی بودن همه ستون‌ها
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    print(f"[هشدار] ستون {col} به عدد تبدیل نشد - حذف می‌شود")
                    columns_to_drop.append(col)
        
        # حذف ستون‌هایی که هنوز object هستند
        object_cols = X.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            print(f"[حذف] ستون‌های object باقیمانده: {object_cols.tolist()}")
            X = X.drop(columns=object_cols)
        
        # جایگزینی مقادیر NaN و Inf
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # بررسی وجود NaN
        if X.isna().any().any():
            print("[هشدار] هنوز NaN وجود دارد - پر کردن با 0")
            X = X.fillna(0)
        
        print(f"\n[اطلاعات] نمونه‌ها: {len(X)}, ویژگی‌ها: {X.shape[1]}")
        print(f"[اطلاعات] توزیع نهایی: عادی(benign)={sum(y==0)}, فیشینگ(phishing)={sum(y==1)}")
        
        return X, y, X.columns.tolist()
    
    # =========================================================
    # دیتاست سوم: ISCX-URL-2016 (فقط فیشینگ و عادی)
    # =========================================================
    elif dataset_name.lower() == 'iscx':
        df = pd.read_csv(file_path)
        
        print(f"[اطلاعات] شکل دیتاست: {df.shape}")
        
        # ستون هدف 'URL_Type_obf_Type' است
        target_col = 'URL_Type_obf_Type'
        
        if target_col not in df.columns:
            raise KeyError(f"ستون '{target_col}' در دیتاست ISCX یافت نشد")
        
        y_raw = df[target_col].copy()
        
        # توزیع اصلی
        print(f"[توزیع اصلی] {y_raw.value_counts().to_dict()}")
        
        # فقط دو کلاس 'phishing' و 'benign' را نگه می‌داریم
        benign_mask = y_raw == 'benign'
        phishing_mask = y_raw == 'phishing'
        
        keep_mask = benign_mask | phishing_mask
        df_filtered = df[keep_mask].copy()
        y_filtered = df_filtered[target_col].copy()
        
        # تبدیل به 0 و 1 (1=phishing, 0=benign)
        y = y_filtered.map(lambda x: 1 if x == 'phishing' else 0)
        
        print(f"[برچسب‌های پس از اصلاح] benign={sum(y==0)}, phishing={sum(y==1)}")
        
        # حذف ستون هدف و ستون‌های غیرعددی
        columns_to_drop = [target_col]
        
        # حذف ستون‌های غیرعددی
        for col in df_filtered.columns:
            if col not in columns_to_drop:
                try:
                    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
                    if df_filtered[col].isna().all():
                        columns_to_drop.append(col)
                        print(f"[حذف] ستون {col} (همه NaN)")
                except:
                    columns_to_drop.append(col)
                    print(f"[حذف] ستون {col} (غیرعددی)")
        
        X = df_filtered.drop(columns=columns_to_drop)
        
        # اطمینان از عددی بودن
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # جایگزینی مقادیر NaN و Inf
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        print(f"\n[اطلاعات] بعد از فیلتر کردن (فقط benign و phishing):")
        print(f"[اطلاعات] نمونه‌ها: {len(X)}, ویژگی‌ها: {X.shape[1]}")
        print(f"[اطلاعات] توزیع نهایی: عادی(benign)={sum(y==0)}, فیشینگ(phishing)={sum(y==1)}")
        
        return X, y, X.columns.tolist()
    
    else:
        raise ValueError(f"دیتاست {dataset_name} پشتیبانی نمی‌شود. گزینه‌ها: PWD2021, phiushiil, iscx")