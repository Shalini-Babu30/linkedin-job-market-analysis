import os
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "postings.csv"
CLEANED_FILE = "linkedin_jobs_cleaned.csv"
ANALYSIS_FILE = "linkedin_jobs_analysis_ready.csv"
VISUALIZATION_DIR = "visualizations"

os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# ==============================
# 1. LOAD DATASET
# ==============================

df = pd.read_csv(DATA_FILE)

print("\n" + "="*60)
print("DATASET OVERVIEW")
print("="*60)

print("Dataset Shape:", df.shape)

print("\nNumber of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])


# ==============================
# 2. COLUMN INFORMATION
# ==============================

print("\n" + "="*60)
print("COLUMN NAMES")
print("="*60)

for i, column in enumerate(df.columns, 1):
    print(i, "->", column)


## ============================================================
# 3. DATA CLEANING
# ============================================================

print("\n" + "="*60)
print("DATA CLEANING")
print("="*60)

# Make a copy so the original dataset remains untouched
clean_df = df.copy()

print("\nOriginal Shape:", clean_df.shape)

# ------------------------------------------------------------
# 3.1 Remove columns with extremely high missing values
# ------------------------------------------------------------

columns_to_drop = [
    "closed_time",
    "skills_desc"
]

clean_df = clean_df.drop(columns=columns_to_drop, errors="ignore")

print("\nDropped columns:", columns_to_drop)
print("Shape after dropping:", clean_df.shape)


# ------------------------------------------------------------
# 3.2 Convert timestamp columns
# ------------------------------------------------------------

clean_df["listed_date"] = pd.to_datetime(
    clean_df["listed_time"],
    unit="ms",
    errors="coerce"
)

clean_df["expiry_date"] = pd.to_datetime(
    clean_df["expiry"],
    unit="ms",
    errors="coerce"
)

clean_df["original_listed_date"] = pd.to_datetime(
    clean_df["original_listed_time"],
    unit="ms",
    errors="coerce"
)


# ------------------------------------------------------------
# 3.3 Create useful date features
# ------------------------------------------------------------

clean_df["year"] = clean_df["listed_date"].dt.year
clean_df["month"] = clean_df["listed_date"].dt.month
clean_df["month_name"] = clean_df["listed_date"].dt.month_name()
clean_df["day_of_week"] = clean_df["listed_date"].dt.day_name()


# ------------------------------------------------------------
# 3.4 Standardize text columns
# ------------------------------------------------------------

text_columns = [
    "company_name",
    "title",
    "location",
    "formatted_work_type",
    "formatted_experience_level",
    "work_type",
    "currency",
    "compensation_type"
]

for column in text_columns:
    if column in clean_df.columns:
        clean_df[column] = clean_df[column].astype("string").str.strip()


# ------------------------------------------------------------
# 3.5 Remove duplicate job IDs
# ------------------------------------------------------------

before = len(clean_df)

clean_df = clean_df.drop_duplicates(
    subset=["job_id"]
)

after = len(clean_df)

print("\nDuplicate job IDs removed:", before - after)


# ------------------------------------------------------------
# 3.6 Check final missing values
# ------------------------------------------------------------

missing = (
    clean_df.isnull()
    .sum()
    .sort_values(ascending=False)
)

print("\nRemaining Missing Values:")
print(missing.head(20))


# ------------------------------------------------------------
# 3.7 Final dataset information
# ------------------------------------------------------------

print("\nFinal Dataset Shape:", clean_df.shape)

print("\nDate Range:")
print("Start:", clean_df["listed_date"].min())
print("End:", clean_df["listed_date"].max())


# ------------------------------------------------------------
# 3.8 Save cleaned dataset
# ------------------------------------------------------------

clean_df.to_csv(
    CLEANED_FILE,
    index=False
)

print("\nCleaned dataset saved successfully!")
# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================

print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)


# ------------------------------------------------------------
# 4.1 Salary Band
# ------------------------------------------------------------

def salary_band(salary):
    if pd.isna(salary):
        return "Not Disclosed"
    elif salary < 50000:
        return "Below $50K"
    elif salary < 100000:
        return "$50K-$100K"
    elif salary < 150000:
        return "$100K-$150K"
    elif salary < 200000:
        return "$150K-$200K"
    else:
        return "$200K+"

clean_df["salary_band"] = clean_df["normalized_salary"].apply(
    salary_band
)


# ------------------------------------------------------------
# 4.2 Remote Work Category
# ------------------------------------------------------------

def work_location_type(row):

    remote = row["remote_allowed"]
    work_type = row["formatted_work_type"]

    if remote == 1:
        return "Remote"

    elif pd.notna(work_type):
        work_type = str(work_type).lower()

        if "hybrid" in work_type:
            return "Hybrid"

        elif "on-site" in work_type or "onsite" in work_type:
            return "On-site"

    return "Not Specified"


clean_df["work_location_type"] = clean_df.apply(
    work_location_type,
    axis=1
)


# ------------------------------------------------------------
# 4.3 Application Rate
# ------------------------------------------------------------

clean_df["application_rate"] = (
    clean_df["applies"]
    .div(clean_df["views"].replace(0, pd.NA))
)

# Avoid infinity values
clean_df["application_rate"] = (
    clean_df["application_rate"]
    .replace([float("inf"), -float("inf")], pd.NA)
)


# ------------------------------------------------------------
# 4.4 Engagement Rate
# ------------------------------------------------------------

clean_df["engagement_rate"] = (
    clean_df["applies"]
    .div(clean_df["views"].replace(0, pd.NA))
    * 100
)


# ------------------------------------------------------------
# 4.5 Job Description Length
# ------------------------------------------------------------

clean_df["description_length"] = (
    clean_df["description"]
    .fillna("")
    .astype(str)
    .str.len()
)


# ------------------------------------------------------------
# 4.6 Job Title Length
# ------------------------------------------------------------

clean_df["title_length"] = (
    clean_df["title"]
    .astype(str)
    .str.len()
)




# ------------------------------------------------------------
# 4.8 Experience Level Cleanup
# ------------------------------------------------------------

clean_df["experience_level"] = (
    clean_df["formatted_experience_level"]
    .fillna("Not Specified")
    .replace({
        "Entry level": "Entry Level",
        "Associate": "Associate",
        "Mid-Senior level": "Mid-Senior",
        "Director": "Director",
        "Executive": "Executive",
        "Internship": "Internship"
    })
)


# ------------------------------------------------------------
# 4.9 Company Job Posting Count
# ------------------------------------------------------------

company_counts = (
    clean_df["company_name"]
    .value_counts()
)

clean_df["company_posting_count"] = (
    clean_df["company_name"]
    .map(company_counts)
)


# ------------------------------------------------------------
# 4.10 Location Job Posting Count
# ------------------------------------------------------------

location_counts = (
    clean_df["location"]
    .value_counts()
)

clean_df["location_posting_count"] = (
    clean_df["location"]
    .map(location_counts)
)
# IMPROVED JOB CATEGORY CLASSIFICATION
# ============================================
def categorize_job(title):

    title = str(title).lower().strip()

    # -------------------------
    # DATA & AI
    # -------------------------

    if any(keyword in title for keyword in [
        "data scientist",
        "data science"
    ]):
        return "Data Scientist"

    elif any(keyword in title for keyword in [
        "data analyst",
        "business analyst",
        "reporting analyst",
        "analytics analyst",
        "business intelligence analyst",
        "bi analyst"
    ]):
        return "Data Analyst"

    elif any(keyword in title for keyword in [
        "data engineer",
        "analytics engineer"
    ]):
        return "Data Engineer"

    elif any(keyword in title for keyword in [
        "machine learning",
        "ml engineer",
        "ai engineer",
        "artificial intelligence",
        "deep learning"
    ]):
        return "AI / ML"


    # -------------------------
    # SOFTWARE DEVELOPMENT
    # -------------------------

    elif any(keyword in title for keyword in [
        "software engineer",
        "software developer",
        "developer",
        "programmer",
        "full stack",
        "frontend",
        "front end",
        "backend",
        "back end",
        "web developer"
    ]):
        return "Software Development"


    # -------------------------
    # PROJECT & PRODUCT
    # -------------------------

    elif any(keyword in title for keyword in [
        "project manager",
        "program manager",
        "project coordinator",
        "program coordinator"
    ]):
        return "Project / Program Management"

    elif any(keyword in title for keyword in [
        "product manager",
        "product owner",
        "product management"
    ]):
        return "Product Management"


    # -------------------------
    # MARKETING
    # -------------------------

    elif any(keyword in title for keyword in [
        "marketing",
        "digital marketing",
        "marketing manager",
        "marketing coordinator",
        "brand manager",
        "marketing specialist",
        "content marketing"
    ]):
        return "Marketing"


    # -------------------------
    # HUMAN RESOURCES
    # -------------------------

    elif any(keyword in title for keyword in [
        "human resources",
        "hr manager",
        "hr coordinator",
        "hr specialist",
        "hr generalist",
        "recruiter",
        "recruiting",
        "talent acquisition",
        "talent manager"
    ]):
        return "Human Resources"


    # -------------------------
    # SALES
    # -------------------------

    elif any(keyword in title for keyword in [
        "sales",
        "account executive",
        "sales representative",
        "sales manager",
        "sales associate",
        "salesperson",
        "business development",
        "business development representative",
        "business development manager"
    ]):
        return "Sales"


    # -------------------------
    # ACCOUNT MANAGEMENT
    # -------------------------

    elif any(keyword in title for keyword in [
        "account manager",
        "global account manager",
        "key account manager",
        "client account manager"
    ]):
        return "Account Management"


    # -------------------------
    # ACCOUNTING & FINANCE
    # -------------------------

    elif any(keyword in title for keyword in [
        "accountant",
        "accounting",
        "senior accountant",
        "staff accountant",
        "accounting manager",
        "controller",
        "bookkeeper",
        "finance",
        "financial analyst",
        "financial manager",
        "accounts payable",
        "accounts receivable",
        "tax accountant",
        "auditor"
    ]):
        return "Accounting / Finance"


    # -------------------------
    # ADMINISTRATION
    # -------------------------

    elif any(keyword in title for keyword in [
        "administrative assistant",
        "administrative",
        "executive assistant",
        "office assistant",
        "office administrator",
        "office manager",
        "receptionist",
        "secretary",
        "clerical",
        "administration"
    ]):
        return "Administration"


    # -------------------------
    # CUSTOMER SERVICE
    # -------------------------

    elif any(keyword in title for keyword in [
        "customer service",
        "customer support",
        "customer care",
        "call center",
        "call centre",
        "client service",
        "support representative",
        "customer success"
    ]):
        return "Customer Service"


    # -------------------------
    # HEALTHCARE
    # -------------------------

    elif any(keyword in title for keyword in [
        "registered nurse",
        "nurse",
        "nursing",
        "medical assistant",
        "medical",
        "healthcare",
        "health care",
        "patient care",
        "pharmacy",
        "pharmacist",
        "therapist",
        "clinical",
        "physician",
        "doctor",
        "dental"
    ]):
        return "Healthcare"


    # -------------------------
    # RETAIL
    # -------------------------

    elif any(keyword in title for keyword in [
        "retail",
        "retail sales",
        "store manager",
        "assistant store manager",
        "store associate",
        "retail associate",
        "merchandiser",
        "cashier"
    ]):
        return "Retail"


    # -------------------------
    # OPERATIONS
    # -------------------------

    elif any(keyword in title for keyword in [
        "operations",
        "operations manager",
        "operations assistant",
        "operations coordinator",
        "operations specialist",
        "operations analyst",
        "warehouse",
        "logistics",
        "supply chain",
        "procurement",
        "inventory"
    ]):
        return "Operations"


    # -------------------------
    # ENGINEERING
    # -------------------------

    elif any(keyword in title for keyword in [
        "electrical engineer",
        "mechanical engineer",
        "civil engineer",
        "project engineer",
        "manufacturing engineer",
        "quality engineer",
        "network engineer",
        "process engineer",
        "industrial engineer",
        "maintenance technician",
        "maintenance engineer",
        "service technician",
        "engineering technician"
    ]):
        return "Engineering"


    # -------------------------
    # LEGAL
    # -------------------------

    elif any(keyword in title for keyword in [
        "attorney",
        "lawyer",
        "legal",
        "paralegal",
        "legal assistant",
        "legal counsel"
    ]):
        return "Legal"


    # -------------------------
    # OTHER
    # -------------------------
    # -------------------------
    # BANKING & FINANCIAL SERVICES
    # -------------------------

    elif any(keyword in title for keyword in [
        "financial advisor",
        "financial adviser",
        "mortgage",
        "loan officer",
        "bank teller",
        "teller",
        "branch manager",
        "banking",
        "bank manager",
        "credit analyst",
        "credit manager",
        "lending",
        "underwriter"
    ]):
        return "Banking / Financial Services"


    # -------------------------
    # EDUCATION
    # -------------------------

    elif any(keyword in title for keyword in [
        "teacher",
        "teaching",
        "educator",
        "professor",
        "instructor",
        "lecturer",
        "special education",
        "school counselor",
        "academic advisor"
    ]):
        return "Education"


    # -------------------------
    # IT & TECHNICAL SUPPORT
    # -------------------------

    elif any(keyword in title for keyword in [
        "devops",
        "technical support",
        "technical specialist",
        "it support",
        "it specialist",
        "it technician",
        "systems administrator",
        "system administrator",
        "network administrator",
        "help desk",
        "desktop support",
        "technical writer"
    ]):
        return "IT / Technical Support"


    # -------------------------
    # CONSTRUCTION
    # -------------------------

    elif any(keyword in title for keyword in [
        "construction",
        "construction superintendent",
        "construction manager",
        "construction supervisor",
        "building superintendent",
        "site superintendent",
        "general contractor"
    ]):
        return "Construction"


    # -------------------------
    # TRANSPORTATION & LOGISTICS
    # -------------------------

    elif any(keyword in title for keyword in [
        "delivery driver",
        "delivery specialist",
        "delivery associate",
        "truck driver",
        "driver",
        "transportation",
        "logistics",
        "warehouse",
        "material handler",
        "shipping",
        "dispatcher",
        "courier"
    ]):
        return "Transportation / Logistics"


    # -------------------------
    # HOSPITALITY & FOOD SERVICE
    # -------------------------

    elif any(keyword in title for keyword in [
        "restaurant",
        "restaurant manager",
        "cook",
        "line cook",
        "chef",
        "dishwasher",
        "server",
        "waiter",
        "waitress",
        "bartender",
        "food service",
        "housekeeper",
        "hotel",
        "hospitality"
    ]):
        return "Hospitality / Food Service"


    # -------------------------
    # DESIGN & CREATIVE
    # -------------------------

    elif any(keyword in title for keyword in [
        "graphic designer",
        "graphic design",
        "designer",
        "creative designer",
        "ux designer",
        "ui designer",
        "visual designer",
        "art director"
    ]):
        return "Design / Creative"


    # -------------------------
    # AUTOMOTIVE
    # -------------------------

    elif any(keyword in title for keyword in [
        "automotive technician",
        "automotive",
        "auto technician",
        "auto detailer",
        "automotive mechanic",
        "auto mechanic"
    ]):
        return "Automotive"


    # -------------------------
    # SCIENCE & LABORATORY
    # -------------------------

    elif any(keyword in title for keyword in [
        "laboratory technician",
        "lab technician",
        "laboratory",
        "lab assistant",
        "research technician",
        "research assistant",
        "chemist",
        "biologist"
    ]):
        return "Science / Laboratory"


    # -------------------------
    # GENERAL MANAGEMENT
    # -------------------------

    elif any(keyword in title for keyword in [
        "general manager",
        "assistant manager",
        "plant manager",
        "branch manager",
        "production supervisor",
        "maintenance supervisor",
        "maintenance manager",
        "supervisor",
        "superintendent"
    ]):
        return "Management"
        # -------------------------
    # BANKING / FINANCIAL SERVICES
    # -------------------------

    elif any(keyword in title for keyword in [
        "financial advisor",
        "tax manager",
        "chief financial officer",
        "relationship banker",
        "loan officer",
        "mortgage",
        "banker",
        "banking"
    ]):
        return "Banking / Financial Services"


    # -------------------------
    # MANAGEMENT
    # -------------------------

    elif any(keyword in title for keyword in [
        "general manager",
        "assistant manager",
        "production manager",
        "property manager",
        "district manager",
        "plant manager",
        "executive director",
        "superintendent",
        "technical lead",
        "management trainee"
    ]):
        return "Management"


    # -------------------------
    # IT / TECHNICAL SUPPORT
    # -------------------------

    elif any(keyword in title for keyword in [
        "system engineer",
        "devops engineer",
        "technical support",
        "technical support specialist",
        "database administrator",
        "solutions architect",
        "technical lead",
        "ct technologist",
        "technologist"
    ]):
        return "IT / Technical Support"


    # -------------------------
    # ENGINEERING - ADDITIONAL
    # -------------------------

    elif any(keyword in title for keyword in [
        "structural engineer",
        "design engineer",
        "controls engineer",
        "automation engineer",
        "mechanical design engineer",
        "automotive engineer",
        "maintenance mechanic",
        "assembler",
        "machine operator"
    ]):
        return "Engineering"


    # -------------------------
    # EDUCATION
    # -------------------------

    elif any(keyword in title for keyword in [
        "teacher",
        "special education",
        "education",
        "instructor",
        "professor",
        "teaching",
        "behavior analyst",
        "bcba"
    ]):
        return "Education"


    # -------------------------
    # SCIENCE / LABORATORY
    # -------------------------

    elif any(keyword in title for keyword in [
        "laboratory",
        "lab technician",
        "laboratory technician",
        "pathologist",
        "phlebotomist",
        "field technician",
        "research"
    ]):
        return "Science / Laboratory"


    # -------------------------
    # DESIGN / CREATIVE
    # -------------------------

    elif any(keyword in title for keyword in [
        "graphic designer",
        "designer",
        "design",
        "copywriter",
        "creative",
        "social media"
    ]):
        return "Design / Creative"


    # -------------------------
    # CONSTRUCTION
    # -------------------------

    elif any(keyword in title for keyword in [
        "construction",
        "construction superintendent",
        "estimator"
    ]):
        return "Construction"


    # -------------------------
    # TRANSPORTATION / LOGISTICS
    # -------------------------

    elif any(keyword in title for keyword in [
        "driver",
        "delivery",
        "shipping",
        "logistics",
        "warehouse",
        "material handler"
    ]):
        return "Transportation / Logistics"


    # -------------------------
    # HOSPITALITY / FOOD SERVICE
    # -------------------------

    elif any(keyword in title for keyword in [
        "cook",
        "line cook",
        "restaurant",
        "server",
        "dishwasher",
        "housekeeper",
        "hospitality"
    ]):
        return "Hospitality / Food Service"


    # -------------------------
    # RETAIL
    # -------------------------

    elif any(keyword in title for keyword in [
        "shopper",
        "merchandise",
        "merchandiser",
        "store assistant"
    ]):
        return "Retail"


    # -------------------------
    # CUSTOMER SERVICE
    # -------------------------

    elif any(keyword in title for keyword in [
        "customer experience",
        "customer experience manager",
        "customer success"
    ]):
        return "Customer Service"


    # -------------------------
    # LEGAL
    # -------------------------

    elif any(keyword in title for keyword in [
        "litigation",
        "litigation associate"
    ]):
        return "Legal"
    # -------------------------
    # VETERINARY / ANIMAL CARE
    # -------------------------

    elif any(keyword in title for keyword in [
        "veterinarian",
        "veterinary",
        "animal care"
    ]):
        return "Veterinary / Animal Care"


    # -------------------------
    # QUALITY / COMPLIANCE
    # -------------------------

    elif any(keyword in title for keyword in [
        "quality assurance",
        "quality specialist",
        "quality manager",
        "quality control",
        "compliance"
    ]):
        return "Quality / Compliance"


    # -------------------------
    # IT / CLOUD
    # -------------------------

    elif any(keyword in title for keyword in [
        "cloud engineer",
        "cloud architect",
        "cloud specialist",
        "devops",
        "system engineer",
        "systems engineer",
        "database administrator",
        "front desk agent"
    ]):
        return "IT / Technical Support"


    # -------------------------
    # CUSTOMER / SOCIAL SERVICES
    # -------------------------

    elif any(keyword in title for keyword in [
        "case manager",
        "behavior technician",
        "registered behavior technician",
        "behavior analyst",
        "customer experience",
        "client experience"
    ]):
        return "Customer Service"


    # -------------------------
    # LOGISTICS / FULFILLMENT
    # -------------------------

    elif any(keyword in title for keyword in [
        "fulfillment associate",
        "fulfillment",
        "material handler",
        "buyer",
        "purchasing",
        "procurement"
    ]):
        return "Transportation / Logistics"


    # -------------------------
    # ADMINISTRATION
    # -------------------------

    elif any(keyword in title for keyword in [
        "front desk",
        "office coordinator",
        "office assistant",
        "administrative",
        "administrative"
    ]):
        return "Administration"


    # -------------------------
    # ENGINEERING / TECHNICIAN
    # -------------------------

    elif any(keyword in title for keyword in [
        "technician",
        "technologist",
        "quality technician",
        "field technician",
        "production technician",
        "automotive technician",
        "maintenance technician"
    ]):
        return "Engineering"


    # -------------------------
    # GENERAL / ENTRY LEVEL
    # -------------------------

    elif any(keyword in title for keyword in [
        "team member",
        "task associate",
        "entry level professional",
        "associate",
        "part time"
    ]):
        return "Other"
        # -------------------------
    # DATA ARCHITECTURE
    # -------------------------

    elif any(keyword in title for keyword in [
        "data architect",
        "data architecture",
        "database architect"
    ]):
        return "Data Engineer"


    # -------------------------
    # HEALTHCARE
    # -------------------------

    elif any(keyword in title for keyword in [
        "dentist",
        "registered nurse",
        "travel rn",
        " rn ",
        "caregiver",
        "patient access",
        "social worker"
    ]):
        return "Healthcare"


    # -------------------------
    # ACCOUNTING / FINANCE
    # -------------------------

    elif any(keyword in title for keyword in [
        "payroll",
        "billing specialist",
        "billing coordinator",
        "billing representative"
    ]):
        return "Accounting / Finance"


    # -------------------------
    # ENGINEERING
    # -------------------------

    elif any(keyword in title for keyword in [
        "application engineer",
        "site reliability engineer",
        "sre",
        "electrician",
        "cnc machinist",
        "test engineer",
        "welder",
        "engineering manager"
    ]):
        return "Engineering"


    # -------------------------
    # OPERATIONS
    # -------------------------

    elif any(keyword in title for keyword in [
        "production associate",
        "production operator",
        "production planner",
        "production supervisor",
        "production manager",
        "material handler"
    ]):
        return "Operations"


    # -------------------------
    # LOGISTICS
    # -------------------------

    elif any(keyword in title for keyword in [
        "forklift operator",
        "forklift",
        "delivery driver",
        "driver",
        "warehouse associate"
    ]):
        return "Transportation / Logistics"


    # -------------------------
    # QUALITY / COMPLIANCE
    # -------------------------

    elif any(keyword in title for keyword in [
        "quality inspector",
        "quality inspection",
        "quality assurance",
        "quality control"
    ]):
        return "Quality / Compliance"


    # -------------------------
    # PROJECT / PROGRAM MANAGEMENT
    # -------------------------

    elif any(keyword in title for keyword in [
        "scrum master",
        "program coordinator",
        "project coordinator"
    ]):
        return "Project / Program Management"


    # -------------------------
    # MANAGEMENT
    # -------------------------

    elif any(keyword in title for keyword in [
        "chief operating officer",
        "coo",
        "service manager",
        "territory manager",
        "community manager",
        "business owner"
    ]):
        return "Management"


    # -------------------------
    # MARKETING / COMMUNICATIONS
    # -------------------------

    elif any(keyword in title for keyword in [
        "communications specialist",
        "content editor",
        "social media"
    ]):
        return "Marketing"


    # -------------------------
    # CONSTRUCTION
    # -------------------------

    elif any(keyword in title for keyword in [
        "laborer",
        "construction laborer",
        "construction worker"
    ]):
        return "Construction"
        # -------------------------
    # HEALTHCARE
    # -------------------------

    elif any(keyword in title for keyword in [
        "registered dietitian",
        "optometrist",
        "occupational therapy assistant",
        "patient service representative",
        "patient care",
        "patient service",
        "cota"
    ]):
        return "Healthcare"


    # -------------------------
    # VETERINARY / ANIMAL CARE
    # -------------------------

    elif any(keyword in title for keyword in [
        "groomer",
        "junior groomer",
        "pet groomer"
    ]):
        return "Veterinary / Animal Care"


    # -------------------------
    # AUTOMOTIVE
    # -------------------------

    elif any(keyword in title for keyword in [
        "mechanic",
        "diesel mechanic",
        "automotive mechanic"
    ]):
        return "Automotive"


    # -------------------------
    # RETAIL
    # -------------------------

    elif any(keyword in title for keyword in [
        "beauty advisor",
        "merchandising service associate",
        "merchandising associate",
        "deli associate",
        "deli production",
        "seasonal merchandising"
    ]):
        return "Retail"


    # -------------------------
    # ENGINEERING
    # -------------------------

    elif any(keyword in title for keyword in [
        "cyber security engineer",
        "project architect",
        "project architecture",
        "engineer"
    ]):
        return "Engineering"


    # -------------------------
    # DATA / TECHNOLOGY
    # -------------------------

    elif any(keyword in title for keyword in [
        "data modeler",
        "data modeling"
    ]):
        return "Data Engineer"


    elif any(keyword in title for keyword in [
        "java architect",
        "software architect",
        "solutions architect"
    ]):
        return "Software Development"


    # -------------------------
    # ADMINISTRATION
    # -------------------------

    elif any(keyword in title for keyword in [
        "contract administrator",
        "data entry specialist",
        "data entry",
        "office expansion",
        "facilities manager"
    ]):
        return "Administration"


    # -------------------------
    # OPERATIONS
    # -------------------------

    elif any(keyword in title for keyword in [
        "continuous improvement manager",
        "equipment operator",
        "shift leader",
        "kitchen leader"
    ]):
        return "Operations"


    # -------------------------
    # QUALITY / COMPLIANCE
    # -------------------------

    elif any(keyword in title for keyword in [
        "safety manager",
        "ehs specialist",
        "environmental health and safety"
    ]):
        return "Quality / Compliance"


    # -------------------------
    # TRANSPORTATION / LOGISTICS
    # -------------------------

    elif any(keyword in title for keyword in [
        "doordash",
        "drive with doordash",
        "dasher"
    ]):
        return "Transportation / Logistics"


    # -------------------------
    # HOSPITALITY / FOOD SERVICE
    # -------------------------

    elif any(keyword in title for keyword in [
        "housekeeping aide",
        "housekeeper",
        "kitchen leader"
    ]):
        return "Hospitality / Food Service"


    # -------------------------
    # MANAGEMENT
    # -------------------------

    elif any(keyword in title for keyword in [
        "chief executive officer",
        "ceo"
    ]):
        return "Management"

    # -------------------------
    # OTHER
    # -------------------------

    else:
        return "Other"
    # Create job category column
clean_df["job_category"] = clean_df["title"].apply(categorize_job)

# ------------------------------------------------------------
# 4.11 Print Feature Summary
# ------------------------------------------------------------

print("\nNew Features Created:")

new_features = [
    "salary_band",
    "work_location_type",
    "application_rate",
    "engagement_rate",
    "description_length",
    "title_length",
    "job_category",
    "experience_level",
    "company_posting_count",
    "location_posting_count"
]

for feature in new_features:
    print(" -", feature)


# ------------------------------------------------------------
# 4.12 Save Final Analytical Dataset
# ------------------------------------------------------------

clean_df.to_csv(
    ANALYSIS_FILE,
    index=False
)

print("\nAnalysis-ready dataset saved successfully!")

print("\nFinal Shape:", clean_df.shape)
# ============================================================
# 5. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "="*60)
print("EXPLORATORY DATA ANALYSIS")
print("="*60)


# ------------------------------------------------------------
# 5.1 JOB CATEGORY DISTRIBUTION
# ------------------------------------------------------------

print("\nTOP JOB CATEGORIES")
print("="*60)

job_category_counts = (
    clean_df["job_category"]
    .value_counts()
)

print(job_category_counts)


# ------------------------------------------------------------
# 5.2 TOP JOB TITLES
# ------------------------------------------------------------

print("\nTOP 20 JOB TITLES")
print("="*60)

top_titles = (
    clean_df["title"]
    .value_counts()
    .head(20)
)

print(top_titles)


# ------------------------------------------------------------
# 5.3 TOP COMPANIES
# ------------------------------------------------------------

print("\nTOP 20 COMPANIES BY JOB POSTINGS")
print("="*60)

top_companies = (
    clean_df["company_name"]
    .value_counts()
    .head(20)
)

print(top_companies)


# ------------------------------------------------------------
# 5.4 TOP LOCATIONS
# ------------------------------------------------------------

print("\nTOP 20 JOB LOCATIONS")
print("="*60)

top_locations = (
    clean_df["location"]
    .value_counts()
    .head(20)
)

print(top_locations)


# ------------------------------------------------------------
# 5.5 EXPERIENCE LEVEL
# ------------------------------------------------------------

print("\nEXPERIENCE LEVEL DISTRIBUTION")
print("="*60)

experience_counts = (
    clean_df["experience_level"]
    .value_counts()
)

print(experience_counts)


# ------------------------------------------------------------
# 5.6 WORK ARRANGEMENT
# ------------------------------------------------------------

print("\nWORK ARRANGEMENT")
print("="*60)

work_arrangement = (
    clean_df["work_location_type"]
    .value_counts()
)

print(work_arrangement)


# ------------------------------------------------------------
# 5.7 SALARY BAND
# ------------------------------------------------------------

print("\nSALARY BAND DISTRIBUTION")
print("="*60)

salary_distribution = (
    clean_df["salary_band"]
    .value_counts()
)

print(salary_distribution)


# ------------------------------------------------------------
# 5.8 SPONSORED VS NON-SPONSORED
# ------------------------------------------------------------

print("\nSPONSORED VS NON-SPONSORED JOBS")
print("="*60)

sponsored_counts = (
    clean_df["sponsored"]
    .value_counts()
)

print(sponsored_counts)


# ------------------------------------------------------------
# 5.9 SUMMARY STATISTICS
# ------------------------------------------------------------

print("\nNUMERICAL SUMMARY")
print("="*60)

print(
    clean_df[
        [
            "views",
            "applies",
            "application_rate",
            "engagement_rate",
            "normalized_salary",
            "description_length"
        ]
    ].describe()
)
# ============================================


# ============================================================
# 5. JOB DEMAND ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("JOB DEMAND ANALYSIS")
print("=" * 60)

# Count job postings by category
job_demand = (
    clean_df["job_category"]
    .value_counts()
    .reset_index()
)

job_demand.columns = ["job_category", "job_postings"]

# Calculate percentage of total postings
job_demand["percentage"] = (
    job_demand["job_postings"]
    / len(clean_df)
    * 100
)

print("\nTOP JOB CATEGORIES BY POSTINGS")
print(job_demand.head(15).to_string(index=False))
# ============================================================
# 6. DATA & AI JOB MARKET ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DATA & AI JOB MARKET ANALYSIS")
print("=" * 60)

data_ai_categories = [
    "Data Analyst",
    "Data Engineer",
    "Data Scientist",
    "AI / ML"
]

data_ai_demand = (
    job_demand[
        job_demand["job_category"].isin(data_ai_categories)
    ]
    .sort_values("job_postings", ascending=False)
)

print("\nDATA & AI JOB DEMAND")
print(data_ai_demand.to_string(index=False))
# ============================================================
# 7. LOCATION ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("LOCATION ANALYSIS")
print("=" * 60)

print("\nLOCATION COLUMNS:")
print([
    col for col in clean_df.columns
    if any(keyword in col.lower()
           for keyword in ["location", "city", "state", "country"])
])
# ============================================================
# 7.1 TOP JOB LOCATIONS
# ============================================================

print("\n" + "=" * 60)
print("TOP JOB LOCATIONS")
print("=" * 60)

top_locations = (
    clean_df["location"]
    .value_counts()
    .head(20)
    .reset_index()
)

top_locations.columns = ["location", "job_postings"]

print(top_locations.to_string(index=False))
# ============================================================
# 7.2 DATA ANALYST LOCATIONS
# ============================================================

print("\n" + "=" * 60)
print("TOP DATA ANALYST LOCATIONS")
print("=" * 60)

data_analyst_locations = (
    clean_df[
        clean_df["job_category"] == "Data Analyst"
    ]["location"]
    .value_counts()
    .head(20)
    .reset_index()
)

data_analyst_locations.columns = [
    "location",
    "data_analyst_postings"
]

print(data_analyst_locations.to_string(index=False))
# ============================================================
# 7.3 WORK LOCATION TYPE
# ============================================================

print("\n" + "=" * 60)
print("WORK LOCATION TYPE")
print("=" * 60)

work_location = (
    clean_df["work_location_type"]
    .value_counts(dropna=False)
    .reset_index()
)

work_location.columns = [
    "work_location_type",
    "job_postings"
]

work_location["percentage"] = (
    work_location["job_postings"]
    / len(clean_df)
    * 100
)

print(work_location.to_string(index=False))
# ============================================================
# 8. EXPERIENCE LEVEL ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("EXPERIENCE LEVEL ANALYSIS")
print("=" * 60)

print("\nEXPERIENCE COLUMNS:")

print([
    col for col in clean_df.columns
    if "experience" in col.lower()
    or "level" in col.lower()
])
# ============================================================
# 8. EXPERIENCE LEVEL ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("EXPERIENCE LEVEL ANALYSIS")
print("=" * 60)

# Overall experience-level distribution
experience_demand = (
    clean_df["formatted_experience_level"]
    .value_counts(dropna=False)
    .reset_index()
)

experience_demand.columns = [
    "experience_level",
    "job_postings"
]

experience_demand["percentage"] = (
    experience_demand["job_postings"]
    / len(clean_df)
    * 100
)

print("\nOVERALL EXPERIENCE LEVEL")
print(experience_demand.to_string(index=False))


# ============================================================
# 8.1 DATA ANALYST EXPERIENCE LEVEL
# ============================================================

data_analyst_experience = (
    clean_df[
        clean_df["job_category"] == "Data Analyst"
    ]["formatted_experience_level"]
    .value_counts(dropna=False)
    .reset_index()
)

data_analyst_experience.columns = [
    "experience_level",
    "data_analyst_postings"
]

print("\n" + "=" * 60)
print("DATA ANALYST EXPERIENCE LEVEL")
print("=" * 60)

print(data_analyst_experience.to_string(index=False))
# ============================================================
# 9. SKILLS ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("SKILLS ANALYSIS")
print("=" * 60)

print("\nSKILL-RELATED COLUMNS:")

print([
    col for col in clean_df.columns
    if any(keyword in col.lower()
           for keyword in [
               "skill",
               "skills",
               "technology",
               "tech",
               "description"
           ])
])
# ============================================================
# 9.1 SKILL DICTIONARY
# ============================================================

# ============================================================
# 9.1 SKILL DICTIONARY
# ============================================================

skills = {

    # -------------------------
    # DATA ANALYTICS
    # -------------------------

    "SQL": [
        "sql",
        "mysql",
        "postgresql",
        "postgres",
        "sql server",
        "t-sql"
    ],

    "Python": [
        "python"
    ],

    "Excel": [
        "microsoft excel",
        "excel"
    ],

    "Power BI": [
        "power bi",
        "powerbi"
    ],

    "Tableau": [
        "tableau"
    ],

    "R": [
    "r programming",
    "r language",
    "rstudio"
    ],

    "Pandas": [
        "pandas"
    ],

    "NumPy": [
        "numpy"
    ],

    "SAS": [
        "sas"
    ],

    "SPSS": [
        "spss"
    ],

    # -------------------------
    # DATA SCIENCE / AI
    # -------------------------

    "Machine Learning": [
        "machine learning",
        "machine-learning"
    ],

    "Deep Learning": [
        "deep learning"
    ],

    "Artificial Intelligence": [
        "artificial intelligence",
        " ai ",
        "ai/ml"
    ],

    "TensorFlow": [
        "tensorflow"
    ],

    "PyTorch": [
        "pytorch"
    ],

    "NLP": [
        "natural language processing",
        "nlp"
    ],

    # -------------------------
    # CLOUD / BIG DATA
    # -------------------------

    "AWS": [
        "aws",
        "amazon web services"
    ],

    "Azure": [
        "azure",
        "microsoft azure"
    ],

    "Google Cloud": [
        "google cloud",
        "gcp"
    ],

    "Spark": [
        "apache spark",
        "spark"
    ],

    "Hadoop": [
        "hadoop"
    ],

    "Databricks": [
        "databricks"
    ],

    # -------------------------
    # PROGRAMMING
    # -------------------------

    "Java": [
        "java"
    ],

    "C++": [
        "c++"
    ],

    "C#": [
        "c#"
    ],

    "JavaScript": [
        "javascript"
    ],

    "HTML/CSS": [
        "html",
        "css"
    ],

    # -------------------------
    # DATABASES
    # -------------------------

    "Oracle": [
        "oracle database",
        "oracle sql"
    ],

    "MongoDB": [
        "mongodb",
        "mongo db"
    ],

    "PostgreSQL": [
        "postgresql",
        "postgres"
    ],

    # -------------------------
    # VERSION CONTROL
    # -------------------------

    "Git": [
        "git",
        "github",
        "gitlab"
    ],

    # -------------------------
    # BUSINESS / OFFICE
    # -------------------------

    "Microsoft Office": [
        "microsoft office",
        "ms office",
        "office suite"
    ],

    "PowerPoint": [
        "powerpoint",
        "microsoft powerpoint"
    ],

    "Project Management": [
        "project management",
        "project planning",
        "project coordination"
    ],

    "Agile": [
        "agile",
        "agile methodology"
    ],

    "Scrum": [
        "scrum",
        "scrum master"
    ],

    # -------------------------
    # MARKETING
    # -------------------------

    "Digital Marketing": [
        "digital marketing",
        "online marketing"
    ],

    "Social Media": [
        "social media",
        "social media marketing"
    ],

    "SEO": [
        "seo",
        "search engine optimization"
    ],

    "Content Marketing": [
        "content marketing",
        "content strategy"
    ],

    "Google Analytics": [
        "google analytics"
    ],

    # -------------------------
    # DESIGN
    # -------------------------

    "Graphic Design": [
        "graphic design",
        "graphic designer"
    ],

    "Adobe Creative Cloud": [
        "adobe creative cloud",
        "creative cloud"
    ],

    "Photoshop": [
        "photoshop",
        "adobe photoshop"
    ],

    "Illustrator": [
        "illustrator",
        "adobe illustrator"
    ],

    "InDesign": [
        "indesign",
        "adobe indesign"
    ],

    # -------------------------
    # COMMUNICATION
    # -------------------------

    "Communication": [
        "communication skills",
        "written communication",
        "verbal communication",
        "communication"
    ],

    "Customer Service": [
        "customer service",
        "customer support",
        "client service"
    ],

    # -------------------------
    # FINANCE
    # -------------------------

    "Financial Analysis": [
        "financial analysis",
        "financial modeling"
    ],

    "Accounting": [
        "accounting",
        "bookkeeping",
        "accounts payable",
        "accounts receivable"
    ]
}
# ============================================================
# 9.2 SKILL EXTRACTION FUNCTION
# ============================================================

def extract_skills(text):

    text = str(text).lower()

    found_skills = []

    for skill, keywords in skills.items():

        for keyword in keywords:

            if keyword in text:
                found_skills.append(skill)
                break

    return found_skills
# ============================================================
# 9.3 SKILL EXTRACTION TEST
# ============================================================

sample_description = clean_df["description"].dropna().iloc[0]

print("\n" + "=" * 60)
print("SKILL EXTRACTION TEST")
print("=" * 60)

print("Detected skills:")
print(extract_skills(sample_description))
# ============================================================
# 9.3 DESCRIPTION CHECK
# ============================================================

sample_description = clean_df["description"].dropna().iloc[0]

print("\n" + "=" * 60)
print("SAMPLE JOB DESCRIPTION")
print("=" * 60)

print(sample_description[:2000])
# ============================================================
# 9.4 EXTRACT SKILLS FROM ALL JOB DESCRIPTIONS
# ============================================================

print("\n" + "=" * 60)
print("EXTRACTING SKILLS FROM ALL JOB DESCRIPTIONS")
print("=" * 60)

clean_df["detected_skills"] = (
    clean_df["description"]
    .fillna("")
    .apply(extract_skills)
)

print("\nSkill extraction completed.")

print("\nSample extracted skills:")
print(
    clean_df[
        ["title", "detected_skills"]
    ]
    .head(10)
    .to_string(index=False)
)
# ============================================================
# 9.5 TOP SKILLS ACROSS ALL JOB POSTINGS
# ============================================================

from collections import Counter

skill_counter = Counter()

for skill_list in clean_df["detected_skills"]:
    skill_counter.update(skill_list)

top_skills = (
    pd.DataFrame(
        skill_counter.items(),
        columns=["skill", "job_postings"]
    )
    .sort_values(
        "job_postings",
        ascending=False
    )
    .reset_index(drop=True)
)

top_skills["percentage"] = (
    top_skills["job_postings"]
    / len(clean_df)
    * 100
)

print("\n" + "=" * 60)
print("TOP SKILLS ACROSS ALL JOB POSTINGS")
print("=" * 60)

print(
    top_skills
    .head(20)
    .to_string(index=False)
)
# ============================================================
# 9.6 DATA ANALYST SKILL ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DATA ANALYST SKILL ANALYSIS")
print("=" * 60)

# Filter Data Analyst jobs
data_analyst_df = clean_df[
    clean_df["job_category"] == "Data Analyst"
].copy()

print(f"\nData Analyst postings: {len(data_analyst_df)}")


# Count skills within Data Analyst postings
data_analyst_skill_counter = Counter()

for skill_list in data_analyst_df["detected_skills"]:
    data_analyst_skill_counter.update(skill_list)


data_analyst_skills = (
    pd.DataFrame(
        data_analyst_skill_counter.items(),
        columns=["skill", "job_postings"]
    )
    .sort_values(
        "job_postings",
        ascending=False
    )
    .reset_index(drop=True)
)


# Calculate percentage of Data Analyst postings
data_analyst_skills["percentage"] = (
    data_analyst_skills["job_postings"]
    / len(data_analyst_df)
    * 100
)


print("\nTOP SKILLS FOR DATA ANALYST JOBS")

print(
    data_analyst_skills
    .head(20)
    .to_string(index=False)
)
# ============================================================
# 10. COMPANY ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("COMPANY ANALYSIS")
print("=" * 60)

print("\nCOMPANY-RELATED COLUMNS:")

print([
    col for col in clean_df.columns
    if any(keyword in col.lower()
           for keyword in [
               "company",
               "employer",
               "organization"
           ])
])
# ============================================================
# 10.1 TOP COMPANIES BY JOB POSTINGS
# ============================================================

print("\n" + "=" * 60)
print("TOP COMPANIES BY JOB POSTINGS")
print("=" * 60)

top_companies = (
    clean_df["company_name"]
    .value_counts()
    .head(20)
    .reset_index()
)

top_companies.columns = [
    "company_name",
    "job_postings"
]

print(
    top_companies
    .to_string(index=False)
)


# ============================================================
# 10.2 TOP COMPANIES HIRING DATA ANALYSTS
# ============================================================

print("\n" + "=" * 60)
print("TOP COMPANIES HIRING DATA ANALYSTS")
print("=" * 60)

data_analyst_companies = (
    clean_df[
        clean_df["job_category"] == "Data Analyst"
    ]["company_name"]
    .value_counts()
    .head(20)
    .reset_index()
)

data_analyst_companies.columns = [
    "company_name",
    "data_analyst_postings"
]

print(
    data_analyst_companies
    .to_string(index=False)
)
# ============================================================
# 11. SALARY ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("SALARY ANALYSIS")
print("=" * 60)

print("\nSALARY-RELATED COLUMNS:")

print([
    col for col in clean_df.columns
    if any(keyword in col.lower()
           for keyword in [
               "salary",
               "min_compensation",
               "max_compensation",
               "compensation"
           ])
])
# ============================================================
# 11.1 SALARY DATA QUALITY
# ============================================================

print("\n" + "=" * 60)
print("SALARY DATA QUALITY")
print("=" * 60)

# Missing salary information
salary_missing = clean_df["normalized_salary"].isna().sum()
salary_available = clean_df["normalized_salary"].notna().sum()

print(f"\nTotal job postings: {len(clean_df):,}")
print(f"Salary available: {salary_available:,}")
print(f"Salary missing: {salary_missing:,}")

print(
    f"Salary coverage: "
    f"{salary_available / len(clean_df) * 100:.2f}%"
)

# Compensation type distribution
print("\n" + "=" * 60)
print("COMPENSATION TYPE")
print("=" * 60)

print(
    clean_df["compensation_type"]
    .value_counts(dropna=False)
    .to_string()
)

# Salary band distribution
print("\n" + "=" * 60)
print("SALARY BAND")
print("=" * 60)

print(
    clean_df["salary_band"]
    .value_counts(dropna=False)
    .head(20)
    .to_string()
)
# ============================================================
# 11.2 OVERALL SALARY ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("OVERALL SALARY ANALYSIS")
print("=" * 60)

salary_df = clean_df[
    clean_df["normalized_salary"].notna()
].copy()

print(f"\nSalary records analyzed: {len(salary_df):,}")

print("\nSalary statistics:")

print(
    salary_df["normalized_salary"]
    .describe()
    .to_string()
)

print("\nKey salary metrics:")

print(
    f"Median salary: "
    f"${salary_df['normalized_salary'].median():,.0f}"
)

print(
    f"Mean salary: "
    f"${salary_df['normalized_salary'].mean():,.0f}"
)

print(
    f"Minimum salary: "
    f"${salary_df['normalized_salary'].min():,.0f}"
)

print(
    f"Maximum salary: "
    f"${salary_df['normalized_salary'].max():,.0f}"
)
# ============================================================
# 11.3 SALARY OUTLIER CHECK
# ============================================================

print("\n" + "=" * 60)
print("SALARY OUTLIER CHECK")
print("=" * 60)

salary_values = salary_df["normalized_salary"]

print("\nSalary percentiles:")

print(
    salary_values
    .quantile([0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00])
    .to_string()
)

print("\nSalaries above $500K:")
print(
    (salary_values > 500000).sum()
)

print("\nSalaries equal to $0:")
print(
    (salary_values == 0).sum()
)

print("\nTop 20 salaries:")

print(
    salary_values
    .sort_values(ascending=False)
    .head(20)
    .to_string(index=False)
)
# ============================================================
# 11.4 CLEAN SALARY DATA
# ============================================================

print("\n" + "=" * 60)
print("CLEANING SALARY DATA")
print("=" * 60)

salary_clean = salary_df[
    (salary_df["normalized_salary"] >= 20000) &
    (salary_df["normalized_salary"] <= 500000)
].copy()

print(f"\nOriginal salary records: {len(salary_df):,}")
print(f"Clean salary records: {len(salary_clean):,}")
print(f"Records removed: {len(salary_df) - len(salary_clean):,}")

print(
    f"Percentage retained: "
    f"{len(salary_clean) / len(salary_df) * 100:.2f}%"
)

print("\nClean salary statistics:")

print(
    f"Median salary: "
    f"${salary_clean['normalized_salary'].median():,.0f}"
)

print(
    f"Mean salary: "
    f"${salary_clean['normalized_salary'].mean():,.0f}"
)

print(
    f"Minimum salary: "
    f"${salary_clean['normalized_salary'].min():,.0f}"
)

print(
    f"Maximum salary: "
    f"${salary_clean['normalized_salary'].max():,.0f}"
)
# ============================================================
# 11.5 DATA & AI SALARY ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("DATA & AI SALARY ANALYSIS")
print("=" * 60)

data_ai_salary = (
    salary_clean[
        salary_clean["job_category"].isin([
            "Data Analyst",
            "Data Engineer",
            "Data Scientist",
            "AI / ML"
        ])
    ]
    .groupby("job_category")["normalized_salary"]
    .agg(
        postings="count",
        median_salary="median",
        mean_salary="mean",
        min_salary="min",
        max_salary="max"
    )
    .sort_values("median_salary", ascending=False)
    .reset_index()
)

print(
    data_ai_salary.to_string(index=False)
)
# ============================================================
# 11.6 DATA ANALYST SALARY BY EXPERIENCE
# ============================================================

print("\n" + "=" * 60)
print("DATA ANALYST SALARY BY EXPERIENCE")
print("=" * 60)

data_analyst_salary_experience = (
    salary_clean[
        salary_clean["job_category"] == "Data Analyst"
    ]
    .groupby("formatted_experience_level")["normalized_salary"]
    .agg(
        postings="count",
        median_salary="median",
        mean_salary="mean",
        min_salary="min",
        max_salary="max"
    )
    .sort_values("median_salary", ascending=False)
    .reset_index()
)

print(
    data_analyst_salary_experience
    .to_string(index=False)
)
# ============================================================
# 12. DATA ANALYST WORK LOCATION
# ============================================================

print("\n" + "=" * 60)
print("DATA ANALYST WORK LOCATION")
print("=" * 60)

data_analyst_work_location = (
    clean_df[
        clean_df["job_category"] == "Data Analyst"
    ]["work_location_type"]
    .value_counts(dropna=False)
    .reset_index()
)

data_analyst_work_location.columns = [
    "work_location_type",
    "job_postings"
]

data_analyst_work_location["percentage"] = (
    data_analyst_work_location["job_postings"]
    / len(data_analyst_df)
    * 100
)

print(
    data_analyst_work_location
    .to_string(index=False)
)
# ============================================================
# 12.1 DATA ANALYST TOP LOCATIONS
# ============================================================

print("\n" + "=" * 60)
print("TOP DATA ANALYST LOCATIONS - FINAL")
print("=" * 60)

print(
    data_analyst_locations
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 12.2 DATA ANALYST TOP COMPANIES
# ============================================================

print("\n" + "=" * 60)
print("TOP DATA ANALYST COMPANIES - FINAL")
print("=" * 60)

print(
    data_analyst_companies
    .head(10)
    .to_string(index=False)
)
# ============================================================
# 13. VISUALIZATIONS
# ============================================================

import matplotlib.pyplot as plt


# ============================================================
# 13.1 JOB DEMAND BY CATEGORY
# ============================================================

# Exclude "Other" because it contains ambiguous job titles
job_demand_chart = (
    job_demand[
        job_demand["job_category"] != "Other"
    ]
    .sort_values("job_postings", ascending=True)
    .copy()
)

plt.figure(figsize=(12, 9))

plt.barh(
    job_demand_chart["job_category"],
    job_demand_chart["job_postings"]
)

plt.xlabel("Number of Job Postings")
plt.ylabel("Job Category")
plt.title("Job Demand by Category")

plt.tight_layout()

plt.savefig(
    "visualizations/job_demand_by_category.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nChart saved: "
    "visualizations/job_demand_by_category.png"
)
# ============================================================
# 13.2 DATA & AI JOB DEMAND
# ============================================================

data_ai_chart = (
    clean_df[
        clean_df["job_category"].isin([
            "Data Analyst",
            "Data Engineer",
            "Data Scientist",
            "AI / ML"
        ])
    ]
    ["job_category"]
    .value_counts()
    .reindex([
        "Data Analyst",
        "Data Engineer",
        "Data Scientist",
        "AI / ML"
    ])
    .reset_index()
)

data_ai_chart.columns = [
    "job_category",
    "job_postings"
]

plt.figure(figsize=(10, 6))

plt.bar(
    data_ai_chart["job_category"],
    data_ai_chart["job_postings"]
)

plt.xlabel("Job Category")
plt.ylabel("Number of Job Postings")
plt.title("Data & AI Job Demand")

plt.tight_layout()

plt.savefig(
    "visualizations/data_ai_job_demand.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nChart saved: "
    "visualizations/data_ai_job_demand.png"
)
# ============================================================
# 13.3 DATA ANALYST SKILLS
# ============================================================

data_analyst_skills_chart = (
    data_analyst_skills
    .head(15)
    .sort_values("percentage", ascending=True)
    .copy()
)

plt.figure(figsize=(12, 8))

plt.barh(
    data_analyst_skills_chart["skill"],
    data_analyst_skills_chart["percentage"]
)

plt.xlabel("Percentage of Data Analyst Job Postings (%)")
plt.ylabel("Skill")
plt.title("Top Skills Mentioned in Data Analyst Job Descriptions")

plt.tight_layout()

plt.savefig(
    "visualizations/data_analyst_skills.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nChart saved: "
    "visualizations/data_analyst_skills.png"
)
# ============================================================
# 13.4 TOP JOB LOCATIONS
# ============================================================

location_chart = (
    clean_df[
        ~clean_df["location"].isin([
            "United States"
        ])
    ]["location"]
    .value_counts()
    .head(15)
    .sort_values(ascending=True)
    .reset_index()
)

location_chart.columns = [
    "location",
    "job_postings"
]

plt.figure(figsize=(12, 8))

plt.barh(
    location_chart["location"],
    location_chart["job_postings"]
)

plt.xlabel("Number of Job Postings")
plt.ylabel("Location")
plt.title("Top Job Locations")

plt.tight_layout()

plt.savefig(
    "visualizations/top_job_locations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nChart saved: "
    "visualizations/top_job_locations.png"
)
# ============================================================
# 13.5 DATA ANALYST SALARY BY EXPERIENCE
# ============================================================

salary_experience_chart = (
    data_analyst_salary_experience
    .sort_values("median_salary", ascending=True)
    .copy()
)

plt.figure(figsize=(10, 6))

plt.barh(
    salary_experience_chart["formatted_experience_level"],
    salary_experience_chart["median_salary"]
)

plt.xlabel("Median Salary ($)")
plt.ylabel("Experience Level")
plt.title("Data Analyst Median Salary by Experience Level")

plt.tight_layout()

plt.savefig(
    "visualizations/data_analyst_salary_by_experience.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nChart saved: "
    "visualizations/data_analyst_salary_by_experience.png"
)
# ============================================================
# 13.6 DATA & AI SALARY COMPARISON
# ============================================================

data_ai_salary_chart = (
    data_ai_salary
    .sort_values("median_salary", ascending=True)
    .copy()
)

plt.figure(figsize=(10, 6))

plt.barh(
    data_ai_salary_chart["job_category"],
    data_ai_salary_chart["median_salary"]
)

plt.xlabel("Median Salary ($)")
plt.ylabel("Job Category")
plt.title("Median Salary Across Data & AI Roles")

plt.tight_layout()

plt.savefig(
    "visualizations/data_ai_salary_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nChart saved: "
    "visualizations/data_ai_salary_comparison.png"
)
# ============================================================
# 13.7 REMOTE WORK COMPARISON
# ============================================================

overall_remote_pct = (
    clean_df["work_location_type"]
    .eq("Remote")
    .mean()
    * 100
)

data_analyst_remote_pct = (
    clean_df[
        clean_df["job_category"] == "Data Analyst"
    ]["work_location_type"]
    .eq("Remote")
    .mean()
    * 100
)

remote_comparison = pd.DataFrame({
    "Group": [
        "All Job Postings",
        "Data Analyst"
    ],
    "Remote Percentage": [
        overall_remote_pct,
        data_analyst_remote_pct
    ]
})

plt.figure(figsize=(9, 6))

plt.bar(
    remote_comparison["Group"],
    remote_comparison["Remote Percentage"]
)

plt.ylabel("Explicitly Remote Postings (%)")
plt.title("Remote Work: Overall Market vs Data Analyst")

plt.tight_layout()

plt.savefig(
    "visualizations/remote_work_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(
    "\nChart saved: "
    "visualizations/remote_work_comparison.png"
)