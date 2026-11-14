import re
from typing import Any
from app.core.engine import run_eligibility, get_csc_info, search_scheme_rag
from app.core.voice import (
    LOCALIZED_TEMPLATES,
    translate_text,
    localize_scheme_matches,
    llm_extract_profile_attributes
)

ORDERED_FIELDS = ["state", "age", "occupation", "income", "gender", "category", "owns_land"]

FIELD_QUESTIONS_EN = {
    "state": "Step 1 of 7: Which state do you live in?",
    "age": "Step 2 of 7: How old are you (in years)?",
    "occupation": "Step 3 of 7: What is your occupation? (farmer, student, street vendor, artisan, unemployed, entrepreneur, or laborer)",
    "income": "Step 4 of 7: What is your approximate annual family income in rupees?",
    "gender": "Step 5 of 7: What is your gender? (male, female, or other)",
    "category": "Step 6 of 7: Which social category do you belong to? (General, OBC, SC, or ST)",
    "owns_land": "Step 7 of 7: Does your household own cultivable agricultural land? (yes or no)"
}

FIELD_KEY_TO_TEMPLATE = {
    "state": "ask_state",
    "age": "ask_age",
    "occupation": "ask_occupation",
    "income": "ask_income",
    "gender": "ask_gender",
    "category": "ask_category",
    "owns_land": "ask_owns_land"
}

FIELD_LABELS_LOCALIZED = {
    "en": {"state": "State", "age": "Age", "occupation": "Occupation", "income": "Annual Income", "gender": "Gender", "category": "Category", "owns_land": "Owns Land"},
    "hi": {"state": "राज्य", "age": "आयु", "occupation": "व्यवसाय", "income": "वार्षिक आय", "gender": "लिंग", "category": "वर्ग", "owns_land": "कृषि भूमि"},
    "ta": {"state": "மாநிலம்", "age": "வயது", "occupation": "தொழில்", "income": "ஆண்டு வருமானம்", "gender": "பாலினம்", "category": "பிரிவு", "owns_land": "விவசாய நிலம்"},
    "kn": {"state": "ರಾಜ್ಯ", "age": "ವಯಸ್ಸು", "occupation": "ಉದ್ಯೋಗ", "income": "ವಾರ್ಷಿಕ ಆದಾಯ", "gender": "ಲಿಂಗ", "category": "ವರ್ಗ", "owns_land": "ಕೃಷಿ ಭೂಮಿ"},
    "pa": {"state": "ਰਾਜ", "age": "ਉਮਰ", "occupation": "ਕਿੱਤਾ", "income": "ਸਾਲਾਨਾ ਆਮਦਨ", "gender": "ਲਿੰਗ", "category": "ਸ਼੍ਰੇਣੀ", "owns_land": "ਖੇਤੀਬਾੜੀ ਜ਼ਮੀਨ"},
    "bn": {"state": "রাজ্য", "age": "বয়স", "occupation": "পেশা", "income": "বার্ষিক আয়", "gender": "লিঙ্গ", "category": "বিভাগ", "owns_land": "কৃষি জমি"}
}

VALUE_TRANSLATIONS = {
    "hi": {"male": "पुरुष", "female": "महिला", "other": "अन्य", "farmer": "किसान", "student": "छात्र", "street_vendor": "रेहड़ी-पटरी विक्रेता", "artisan": "कारीगर", "unemployed": "बेरोजगार / गृहिणी", "entrepreneur": "उद्यमी", "laborer": "श्रमिक", "True": "हाँ", "False": "नहीं"},
    "ta": {"male": "ஆண்", "female": "பெண்", "other": "மற்றவை", "farmer": "விவசாயி", "student": "மாணவர்", "street_vendor": "தெரு வியாபாரி", "artisan": "கைவினைஞர்", "unemployed": "வேலையில்லாதவர் / இல்லத்தரசி", "entrepreneur": "தொழில்முனைவோர்", "laborer": "தொழிலாளி", "True": "ஆம்", "False": "இல்லை"},
    "kn": {"male": "ಪುರುಷ", "female": "ಮಹಿಳೆ", "other": "ಇತರೆ", "farmer": "ರೈತ", "student": "ವಿದ್ಯಾರ್ಥಿ", "street_vendor": "ಬೀದಿ ವ್ಯಾಪಾರಿ", "artisan": "ಕುಶಲಕರ್ಮಿ", "unemployed": "ನಿರುದ್ಯೋಗಿ / ಗೃಹಿಣಿ", "entrepreneur": "ಉದ್ಯಮಿ", "laborer": "ಕಾರ್ಮಿಕ", "True": "ಹೌದು", "False": "ಇಲ್ಲ"},
    "pa": {"male": "ਪੁਰਸ਼", "female": "ਮਹਿਲਾ", "other": "ਹੋਰ", "farmer": "ਕਿਸਾਨ", "student": "ਵਿਦਿਆਰਥੀ", "street_vendor": "ਰੇਹੜੀ ਫੜ੍ਹੀ ਵਾਲਾ", "artisan": "ਕਾਰੀਗਰ", "unemployed": "ਬੇਰੁਜ਼ਗਾਰ / ਘਰੇਲੂ ਔਰਤ", "entrepreneur": "ਉੱਦਮੀ", "laborer": "ਮਜ਼ਦੂਰ", "True": "ਹਾਂ", "False": "ਨਹੀਂ"},
    "bn": {"male": "পুরুষ", "female": "মহিলা", "other": "অন্যান্য", "farmer": "কৃষক", "student": "ছাত্র/ছাত্রী", "street_vendor": "পথ ব্যবসায়ী", "artisan": "কারিগর", "unemployed": "বেকার / গৃহিণী", "entrepreneur": "উদ্যোক্তা", "laborer": "শ্রমিক", "True": "হ্যাঁ", "False": "না"}
}

INDIAN_STATES = {
    "karnataka": "Karnataka", "bangalore": "Karnataka", "bengaluru": "Karnataka", "mysuru": "Karnataka", "mysore": "Karnataka", "hubli": "Karnataka", "mangalore": "Karnataka",
    "कर्नाटक": "Karnataka", "ಕರ್ನಾಟಕ": "Karnataka", "ಕರ್ನಾಟಕದಲ್ಲಿ": "Karnataka", "கர்நாடகா": "Karnataka", "কর্ণাটক": "Karnataka", "ਕਰਨਾਟਕ": "Karnataka",
    "tamil nadu": "Tamil Nadu", "tamilnadu": "Tamil Nadu", "chennai": "Tamil Nadu", "coimbatore": "Tamil Nadu", "madurai": "Tamil Nadu", "tn": "Tamil Nadu",
    "तमिलनाडु": "Tamil Nadu", "தமிழ்நாடு": "Tamil Nadu", "தமிழ்நாட்டில்": "Tamil Nadu", "தமிழ்நாட்டைச்": "Tamil Nadu", "ತಮಿಳುನಾಡು": "Tamil Nadu", "তামিলনাড়ু": "Tamil Nadu", "ਤਮਿਲਨਾਡੂ": "Tamil Nadu",
    "west bengal": "West Bengal", "bengal": "West Bengal", "kolkata": "West Bengal", "howrah": "West Bengal", "wb": "West Bengal",
    "पश्चिम बंगाल": "West Bengal", "बंगाल": "West Bengal", "পশ্চিমবঙ্গ": "West Bengal", "পশ্চিমবঙ্গের": "West Bengal", "বাংলা": "West Bengal", "ಪಶ್ಚಿಮ ಬಂಗಾಳ": "West Bengal", "மேற்கு வங்காளம்": "West Bengal", "ਪੱਛਮੀ ਬੰਗਾਲ": "West Bengal",
    "maharashtra": "Maharashtra", "mumbai": "Maharashtra", "pune": "Maharashtra", "nagpur": "Maharashtra", "mh": "Maharashtra",
    "महाराष्ट्र": "Maharashtra", "ಮಹಾರಾಷ್ಟ್ರ": "Maharashtra", "மகாராஷ்டிரா": "Maharashtra", "মহারাষ্ট্র": "Maharashtra", "ਮਹਾਰਾਸ਼ਟਰ": "Maharashtra",
    "madhya pradesh": "Madhya Pradesh", "bhopal": "Madhya Pradesh", "indore": "Madhya Pradesh", "मध्य प्रदेश": "Madhya Pradesh", "mp": "Madhya Pradesh",
    "uttar pradesh": "Uttar Pradesh", "lucknow": "Uttar Pradesh", "kanpur": "Uttar Pradesh", "varanasi": "Uttar Pradesh", "noida": "Uttar Pradesh", "उत्तर प्रदेश": "Uttar Pradesh", "यूपी": "Uttar Pradesh", "up": "Uttar Pradesh", "ಉತ್ತರ ಪ್ರದೇಶ": "Uttar Pradesh", "உத்தரப் பிரதேசம்": "Uttar Pradesh", "উত্তর প্রদেশ": "Uttar Pradesh", "ਉੱਤਰ ਪ੍ਰਦੇਸ਼": "Uttar Pradesh",
    "telangana": "Telangana", "hyderabad": "Telangana", "तेलंगाना": "Telangana", "ತೆಲಂಗಾಣ": "Telangana", "தெலுங்கானா": "Telangana", "তেলেঙ্গানা": "Telangana",
    "odisha": "Odisha", "orissa": "Odisha", "bhubaneswar": "Odisha", "ओडिशा": "Odisha", "उड़ीसा": "Odisha", "ஒடிசா": "Odisha", "ಒಡಿಶಾ": "Odisha", "ওড়িশা": "Odisha",
    "rajasthan": "Rajasthan", "jaipur": "Rajasthan", "राजस्थान": "Rajasthan", "ராஜஸ்தான்": "Rajasthan", "ರಾಜಸ್ಥಾನ": "Rajasthan", "রাজস্থান": "Rajasthan", "ਰਾਜਸਥਾਨ": "Rajasthan",
    "punjab": "Punjab", "ludhiana": "Punjab", "amritsar": "Punjab", "jalandhar": "Punjab", "patiala": "Punjab", "पंजाब": "Punjab", "ਪੰਜਾਬ": "Punjab", "pb": "Punjab", "பஞ்சாப்": "Punjab", "ಪಂಜಾಬ್": "Punjab", "পাঞ্জাব": "Punjab",
    "delhi": "Delhi", "new delhi": "Delhi", "दिल्ली": "Delhi", "டெல்லி": "Delhi", "ದೆಹಲಿ": "Delhi", "দিল্লি": "Delhi", "ਦਿੱਲੀ": "Delhi",
    "bihar": "Bihar", "patna": "Bihar", "बिहार": "Bihar", "gujarat": "Gujarat", "गुजरात": "Gujarat", "kerala": "Kerala", "केरल": "Kerala", "andhra pradesh": "Andhra Pradesh", "haryana": "Haryana", "हरियाणा": "Haryana", "assam": "Assam"
}

OCCUPATION_MAP = {
    "farmer": "farmer", "kisan": "farmer", "agriculture": "farmer", "cultivator": "farmer", "farming": "farmer", "krishi": "farmer",
    "किसान": "farmer", "खेती": "farmer", "कृषि": "farmer", "விவசாயி": "farmer", "விவசாயம்": "farmer", "ರೈತ": "farmer", "ಕೃಷಿ": "farmer", "ਕਿਸਾਨ": "farmer", "ਖੇਤੀ": "farmer", "কৃষক": "farmer", "চাষী": "farmer",
    "student": "student", "college": "student", "school": "student", "studying": "student", "study": "student", "university": "student", "btech": "student", "engineering": "student", "degree": "student", "diploma": "student", "child": "student", "kid": "student",
    "छात्र": "student", "छात्रा": "student", "पढ़ाई": "student", "विद्यार्थी": "student", "स्टूडेंट": "student", "बच्चा": "student", "மாணவர்": "student", "மாணவி": "student", "ವಿದ್ಯಾರ್ಥಿ": "student", "ਵਿਦਿਆਰਥੀ": "student", "ছাত্র": "student", "ছাত্রী": "student", "পড়ুয়া": "student",
    "street vendor": "street_vendor", "street_vendor": "street_vendor", "vendor": "street_vendor", "hawker": "street_vendor", "thela": "street_vendor", "vegetable seller": "street_vendor", "fruit seller": "street_vendor", "tea stall": "street_vendor",
    "रेहड़ी": "street_vendor", "पटरी": "street_vendor", "ठेला": "street_vendor", "फेरीवाला": "street_vendor", "दुकानदार": "street_vendor", "வியாபாரி": "street_vendor", "ಬೀದಿ ವ್ಯಾಪಾರಿ": "street_vendor", "ਰੇਹੜੀ": "street_vendor", "ਦੁਕਾਨਦਾਰ": "street_vendor", "হকার": "street_vendor", "ব্যবসায়ী": "street_vendor",
    "artisan": "artisan", "carpenter": "artisan", "tailor": "artisan", "potter": "artisan", "blacksmith": "artisan", "cobbler": "artisan", "mason": "artisan", "barber": "artisan", "goldsmith": "artisan", "weaver": "artisan", "vishwakarma": "artisan",
    "कारीगर": "artisan", "बढ़ई": "artisan", "दर्जी": "artisan", "कुम्हार": "artisan", "लोहार": "artisan", "नाई": "artisan", "राजमिस्त्री": "artisan", "கைவினைஞர்": "artisan", "ಕುಶಲಕರ್ಮಿ": "artisan", "ਕਾਰੀਗਰ": "artisan", "কারিগর": "artisan",
    "unemployed": "unemployed", "jobless": "unemployed", "homemaker": "unemployed", "housewife": "unemployed", "no job": "unemployed", "not working": "unemployed", "none": "unemployed",
    "बेरोजगार": "unemployed", "गृहिणी": "unemployed", "हाउसवाइफ": "unemployed", "ನಿರುದ್ಯೋಗಿ": "unemployed", "ಗೃಹಿಣಿ": "unemployed", "வேலையில்லாதவர்": "unemployed", "இல்லத்தரசி": "unemployed", "ਬੇਰੁਜ਼ਗਾਰ": "unemployed", "ਘਰੇਲੂ ਔਰਤ": "unemployed", "বেকার": "unemployed", "গৃহিণী": "unemployed",
    "entrepreneur": "entrepreneur", "business": "entrepreneur", "startup": "entrepreneur", "shopkeeper": "shopkeeper", "self employed": "self_employed", "self_employed": "self_employed",
    "laborer": "laborer", "labour": "laborer", "labourer": "laborer", "worker": "laborer", "daily wage": "laborer", "construction worker": "laborer", "driver": "laborer", "rickshaw": "laborer", "domestic worker": "laborer",
    "मजदूर": "laborer", "श्रमिक": "laborer", "रिक्शा": "laborer", "கூலி": "laborer", "தொழிலாளி": "laborer", "ಕಾರ್ಮಿಕ": "laborer", "ਮਜ਼ਦੂਰ": "laborer", "শ্রমিক": "laborer"
}

OUT_OF_SCOPE_PATTERNS = [
    r"(us|usa|uk|canada|schengen|tourist|student|work)\s+visa",
    r"passport", r"bitcoin", r"crypto", r"stock\s+market",
    r"weather", r"cricket", r"ipl", r"movie", r"flight\s+ticket"
]

def is_out_of_scope_query(text: str) -> bool:
    lower = (text or "").lower()
    for pat in OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, lower):
            return True
    return False

INDIC_DIGITS = str.maketrans(
    "०१२३४५६७८९০১২৩৪৫৬৭৮৯೦೧೨೩೪೫೬೭೮೯੦੧੨੩੪੫੬੭੮੯௦௧௨௩௪௫௬௭௮௯",
    "01234567890123456789012345678901234567890123456789"
)

def convert_all_spoken_words_to_numbers(text: str) -> str:
    out = text.translate(INDIC_DIGITS).lower()
    out = re.sub(r"[-]", " ", out)

    tens_map = {
        "twenty": 20, "thirty": 30, "forty": 40, "fourty": 40,
        "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
    }
    units_map = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9
    }
    for t_word, t_val in tens_map.items():
        for u_word, u_val in units_map.items():
            out = re.sub(rf"{t_word}\s+{u_word}", str(t_val + u_val), out)

    single_nums = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
        "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
        "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
        "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19",
        "twenty": "20", "thirty": "30", "forty": "40", "fourty": "40", "fifty": "50",
        "sixty": "60", "seventy": "70", "eighty": "80", "ninety": "90", "hundred": "100",
        # Hindi 1-90
        "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पांच": "5", "पाँच": "5", "छह": "6", "सात": "7", "आठ": "8", "नौ": "9", "दस": "10",
        "ग्यारह": "11", "बारह": "12", "तेरह": "13", "चौदह": "14", "पंद्रह": "15",
        "सोलह": "16", "सत्रह": "17", "अठारह": "18", "उन्नीस": "19", "बीस": "20",
        "इक्कीस": "21", "बाईस": "22", "तेईस": "23", "चौबीस": "24", "पच्चीस": "25",
        "छब्बीस": "26", "सत्ताईस": "27", "अट्ठाईस": "28", "उनतीस": "29", "तीस": "30",
        "इकतीस": "31", "बत्तीस": "32", "तैंतीस": "33", "चौंतीस": "34", "पैंतीस": "35",
        "छत्तीस": "36", "सैंतीस": "37", "अड़तीस": "38", "उनतालीस": "39", "चालीस": "40",
        "पैंतालीस": "45", "पचास": "50", "पचपन": "55", "साठ": "60", "पैंसठ": "65", "सत्तर": "70", "अस्सी": "80", "नब्बे": "90",
        "ek": "1", "do": "2", "teen": "3", "chaar": "4", "paanch": "5", "saat": "7", "aath": "8", "das": "10",
        "bees": "20", "pachees": "25", "tees": "30", "chalis": "40", "pachas": "50", "saath": "60",
        # Tamil, Kannada, Punjabi, Bengali numbers
        "ஐந்து": "5", "ஏழு": "7", "பத்து": "10", "இருபது": "20", "முப்பது": "30", "நாற்பது": "40", "ஐம்பது": "50", "அறுபது": "60",
        "ಐದು": "5", "ಏಳು": "7", "ಹತ್ತು": "10", "ಇಪ್ಪತ್ತು": "20", "ಮೂವತ್ತು": "30", "ನಲವತ್ತು": "40", "ಐವತ್ತು": "50", "ಅರವತ್ತು": "60",
        "ਪੰਜ": "5", "ਸੱਤ": "7", "ਦਸ": "10", "ਵੀਹ": "20", "ਤੀਹ": "30", "ਚਾਲੀ": "40", "ਪੰਜਾਹ": "50", "ਸੱਠ": "60",
        "পাঁচ": "5", "সাত": "7", "দশ": "10", "কুড়ি": "20", "ত্রিশ": "30", "চল্লিশ": "40", "পঞ্চাশ": "50", "ষাট": "60"
    }
    for word, digit in sorted(single_nums.items(), key=lambda x: len(x[0]), reverse=True):
        out = re.sub(rf"(?<!\w){re.escape(word)}(?!\w)", f" {digit} ", out)

    def _mult_thousand(m):
        return f" {int(float(m.group(1)) * 1000)} "
    out = re.sub(r"(\d+(?:\.\d+)?)\s*(?:k|thousand|hazaar|hazar|हजार|हज़ार|ஆயிரம்|ಸಾವಿರ|ਹਜ਼ਾਰ|হাজার)", _mult_thousand, out)

    def _mult_lakh(m):
        return f" {int(float(m.group(1)) * 100000)} "
    out = re.sub(r"(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs|लाख|லட்சம்|ಲಕ್ಷ|ਲੱਖ|লক্ষ|লাখ)", _mult_lakh, out)

    return out

def next_question_key(profile: dict[str, Any]) -> str | None:
    """Strictly returns the first missing attribute in the 7-step sequence."""
    for key in ORDERED_FIELDS:
        if profile.get(key) is None:
            return key
    return None

def extract_profile_from_text(text: str, current_profile: dict[str, Any], last_asked_field: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    if not text:
        return current_profile, {}

    # Determine which step we are currently trying to fill
    active_step = last_asked_field or next_question_key(current_profile)
    lower = convert_all_spoken_words_to_numbers(text)
    extracted: dict[str, Any] = {}

    # Find ALL digit sequences anywhere in the string (even if attached to Unicode characters!)
    raw_num_strings = re.findall(r"\d+(?:,\d+)*", lower)
    all_numbers = []
    for s in raw_num_strings:
        try:
            all_numbers.append(int(s.replace(",", "")))
        except Exception:
            pass

    # ==========================================================================
    # PRIORITY 1: DIRECT STEP-BY-STEP CAPTURE FOR THE CURRENT ACTIVE QUESTION!
    # ==========================================================================
    if active_step == "age":
        # Accept ANY age from 0 to 120! (e.g. "7 years", "45 years", "0", "3", "85")
        for num in all_numbers:
            if 0 <= num <= 120:
                extracted["age"] = num
                break
        if "age" not in extracted and any(w in lower for w in ["baby", "infant", "newborn", "शिशु", "नवजात"]):
            extracted["age"] = 1

    elif active_step == "income":
        monthly_match = re.search(r"(\d[\d,]*)\s*(?:per month|a month|monthly|pm|mahina|महीना|महीने|மாதம்|ತಿಂಗಳಿಗೆ|ਮਹੀਨਾ|মাসে)", lower)
        if monthly_match:
            extracted["income"] = int(monthly_match.group(1).replace(",", "")) * 12
        elif any(w in lower for w in ["zero", "no income", "nothing", "nil", "none", "शून्य", "कुछ नहीं"]):
            extracted["income"] = 0
        elif all_numbers:
            v = all_numbers[-1]
            # If user says "80" or "90" during the income step, treat 1..499 as thousands (80,000), or 0 as 0
            extracted["income"] = v * 1000 if 0 < v < 500 else v

    elif active_step == "owns_land":
        if any(w in lower for w in ["no", "nope", "false", "nahi", "na", "dont", "don't", "landless", "ना", "नहीं", "भूमिहीन", "இல்லை", "ಇಲ್ಲ", "ਨਹੀਂ", "না"]):
            extracted["owns_land"] = False
        elif any(w in lower for w in ["yes", "yeah", "yep", "true", "haan", "ha", "own", "have", "हाँ", "जी", "ஆம்", "ஆமா", "ಹೌದು", "ਹਾਂ", "ਜੀ", "হ্যাঁ"]):
            extracted["owns_land"] = True

    # ==========================================================================
    # PRIORITY 2: GENERAL ENTITY EXTRACTION (State, Occupation, Gender, Category, etc.)
    # ==========================================================================
    for kw, canonical in sorted(INDIAN_STATES.items(), key=lambda x: len(x[0]), reverse=True):
        if kw in lower:
            if len(kw) <= 3:
                if re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", lower):
                    extracted["state"] = canonical
                    break
            else:
                extracted["state"] = canonical
                break

    for kw, canonical in sorted(OCCUPATION_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        if kw in lower:
            if re.search(rf"(not|no|never)\s+(?:a\s+|an\s+)?{re.escape(kw)}", lower):
                continue
            extracted["occupation"] = canonical
            break

    if any(w in lower for w in ["female", "woman", "women", "girl", "lady", "mother", "wife", "widow", "महिला", "लड़की", "औरत", "गृहिणी", "छात्रा", "பெண்", "மாணவி", "ಮಹಿಳೆ", "ಗೃಹಿಣಿ", "ਮਹਿਲਾ", "ਔਰਤ", "মহিলা", "গৃহিণী", "ছাত্রী"]):
        extracted["gender"] = "female"
    elif any(w in lower for w in ["male", "man", "men", "boy", "father", "husband", "पुरुष", "लड़का", "आदमी", "ஆண்", "ಪುರುಷ", "ਪੁਰਸ਼", "পুরুষ"]):
        extracted["gender"] = "male"
    elif active_step == "gender" and any(w in lower for w in ["other", "transgender", "अन्य", "மற்றவை", "ಇತರೆ", "ਹੋਰ", "অন্যান্য"]):
        extracted["gender"] = "other"

    if re.search(r"(?<!\w)(sc|scheduled caste|dalit)(?!\w)", lower) or any(w in lower for w in ["अनुसूचित जाति", "एस सी", "एससी", "எஸ்சி", "ಎಸ್‌ಸಿ", "ਐਸਸੀ", "এসসি"]):
        extracted["category"] = "SC"
    elif re.search(r"(?<!\w)(st|scheduled tribe|adivasi|tribal)(?!\w)", lower) or any(w in lower for w in ["अनुसूचित जनजाति", "एस टी", "एसटी", "எஸ்டி", "ಎಸ್‌ಟಿ", "ਐਸਟੀ", "এসটি"]):
        extracted["category"] = "ST"
    elif re.search(r"(?<!\w)(obc|other backward|backward class|bc|ebc)(?!\w)", lower) or any(w in lower for w in ["अन्य पिछड़ा", "ओबीसी", "ओ बी सी", "ஓபிசி", "ಒಬಿಸಿ", "ਓਬੀਸੀ", "ওবিসি"]):
        extracted["category"] = "OBC"
    elif re.search(r"(?<!\w)(general|unreserved|open category|forward|gen)(?!\w)", lower) or any(w in lower for w in ["सामान्य", "जनरल", "பொது", "ಸಾಮಾನ್ಯ", "ਜਨਰਲ", "সাধারণ", "জেনারেল"]):
        extracted["category"] = "General"

    if "owns_land" not in extracted:
        if any(p in lower for p in ["no land", "landless", "don't own land", "do not own land", "dont have land", "without land", "जमीन नहीं", "भूमिहीन", "खेत नहीं", "நிலம் இல்லை", "ಭೂಮಿ ಇಲ್ಲ", "ਜ਼ਮੀਨ ਨਹੀਂ", "জমি নেই"]):
            extracted["owns_land"] = False
        elif any(p in lower for p in ["own land", "own agricultural land", "have land", "cultivable land", "acres", "acre", "जमीन है", "खेत है", "நிலம் உள்ளது", "ಭೂಮಿ ಇದೆ", "ਜ਼ਮੀਨ ਹੈ", "জমি আছে"]):
            extracted["owns_land"] = True

    # Multi-fact sentence age & income extraction (when user says "I am 7 years old from Karnataka and income is 90000")
    if "age" not in extracted:
        age_m1 = re.search(r"(\d{1,3})\s*(?:years?\s*old|yrs?\s*old|years?|yrs?|yo|साल|वर्ष|उम्र|வயது|ವರ್ಷದ|ವರ್ಷ|ਸਾਲ|বছর|বয়সী)", lower)
        age_m2 = re.search(r"(?:age|aged|i\s*am|im|i'm|उम्र|आयु|வயது|ವಯಸ್ಸು|ਉਮਰ|বয়স)[^0-9]{0,12}(\d{1,3})", lower)
        if age_m1 and 0 <= int(age_m1.group(1)) <= 120:
            extracted["age"] = int(age_m1.group(1))
        elif age_m2 and 0 <= int(age_m2.group(1)) <= 120:
            extracted["age"] = int(age_m2.group(1))
        elif current_profile.get("age") is None and active_step != "income":
            for num in all_numbers:
                if 0 <= num <= 120:
                    extracted["age"] = num
                    break

    if "income" not in extracted:
        for num in all_numbers:
            if num >= 1000 and num != extracted.get("age"):
                extracted["income"] = num
                break

    # LLM NLU fallback for any remaining unrecognized regional phrasing
    if active_step not in extracted and len(text.strip()) > 1:
        llm_ext = llm_extract_profile_attributes(text, current_profile, active_step)
        for k, v in llm_ext.items():
            if k in ORDERED_FIELDS and v is not None and k not in extracted:
                extracted[k] = v

    updated = dict(current_profile)
    for k, v in extracted.items():
        if v is not None:
            updated[k] = v

    return updated, extracted

def format_localized_confirmation(extracted: dict[str, Any], language: str) -> str:
    if not extracted:
        return ""
    lang = language if language in FIELD_LABELS_LOCALIZED else "en"
    labels = FIELD_LABELS_LOCALIZED[lang]
    val_map = VALUE_TRANSLATIONS.get(lang, {})

    parts = []
    for k in ORDERED_FIELDS:
        if k in extracted and extracted[k] is not None:
            raw_v = extracted[k]
            if k == "income":
                disp_v = f"₹{int(raw_v):,}"
            elif k == "owns_land":
                disp_v = val_map.get(str(bool(raw_v)), "Yes" if raw_v else "No")
            else:
                disp_v = val_map.get(str(raw_v), str(raw_v))
                if k == "state" and lang != "en":
                    disp_v = translate_text(str(raw_v), "en", lang)
            parts.append(f"{labels[k]}: {disp_v}")

    if lang == "en":
        return "Noted (" + ", ".join(parts) + "). "
    prefix = LOCALIZED_TEMPLATES.get(lang, {}).get("confirmed_prefix", "Noted: ")
    return f"{prefix}({', '.join(parts)})। " if lang in ("hi", "pa", "bn") else f"{prefix}({', '.join(parts)}). "

def make_response(
    profile: dict[str, Any],
    message: str = "",
    language: str = "en",
    last_asked_field: str | None = None,
    asked_counts: dict[str, int] | None = None
) -> dict[str, Any]:
    if asked_counts is None:
        asked_counts = {}
    disclaimer_en = (
        "Unofficial AI Helper (Not an official government service). "
        "Scheme rules can change; always verify details on the official myScheme portal. "
        "We never ask for Aadhaar numbers, bank details, or OTPs."
    )
    disclaimer = translate_text(disclaimer_en, "en", language) if language != "en" else disclaimer_en

    if is_out_of_scope_query(message):
        en_refusal = (
            "I cannot answer this question because it is outside our official government welfare scheme "
            "knowledge base (myScheme). I do not guess or invent information. Ask me about Indian central "
            "or state welfare schemes!"
        )
        loc_refusal = LOCALIZED_TEMPLATES.get(language, {}).get("out_of_scope") or (
            translate_text(en_refusal, "en", language) if language != "en" else en_refusal
        )
        return {
            "reply": loc_refusal,
            "english_reply": en_refusal,
            "stage": "out_of_scope_refusal",
            "profile_complete": False,
            "next_field": last_asked_field,
            "extracted_this_turn": {},
            "matches": [],
            "rejected_matches": [],
            "rag_chunks": [],
            "csc_guidance": get_csc_info(profile.get("state"), language),
            "disclaimer": disclaimer
        }

    # Always determine the active question BEFORE extraction so even if last_asked_field was None, it knows which step we are on!
    current_step_field = last_asked_field or next_question_key(profile)
    updated_profile, extracted_now = extract_profile_from_text(message, profile, current_step_field)
    profile.update(updated_profile)

    q_key = next_question_key(profile)
    confirm_en = format_localized_confirmation(extracted_now, "en")
    confirm_loc = format_localized_confirmation(extracted_now, language)

    if q_key:
        step_num = ORDERED_FIELDS.index(q_key) + 1
        en_question = FIELD_QUESTIONS_EN[q_key]
        en_reply = f"{confirm_en}{en_question}"

        partial_results = run_eligibility(profile)
        visible_partial = localize_scheme_matches(
            [x for x in partial_results if x["status"] != "not_eligible"], language
        )
        rejected_partial = localize_scheme_matches(
            [x for x in partial_results if x["status"] == "not_eligible"][:10], language
        )

        if language != "en" and language in LOCALIZED_TEMPLATES:
            tmpl = LOCALIZED_TEMPLATES[language]
            loc_q = tmpl.get(FIELD_KEY_TO_TEMPLATE[q_key], en_question)
            step_word = tmpl.get("step_counter", f"Step {step_num}/7")
            loc_reply = f"{confirm_loc}[{step_word.format(n=step_num)}] {loc_q}"
        elif language != "en":
            loc_reply = translate_text(en_reply, "en", language)
        else:
            loc_reply = en_reply

        return {
            "reply": loc_reply,
            "english_reply": en_reply,
            "stage": "collecting_profile",
            "profile_complete": False,
            "next_field": q_key,
            "current_step": step_num,
            "total_steps": len(ORDERED_FIELDS),
            "extracted_this_turn": extracted_now,
            "matches": visible_partial,
            "rejected_matches": rejected_partial,
            "rag_chunks": [],
            "csc_guidance": get_csc_info(profile.get("state"), language),
            "disclaimer": disclaimer
        }

    # All 7 steps complete!
    results = run_eligibility(profile)
    eligible_raw = [x for x in results if x["status"] == "eligible"]
    possibly_raw = [x for x in results if x["status"] == "possibly_eligible"]
    rejected_raw = [x for x in results if x["status"] == "not_eligible"]

    visible_localized = localize_scheme_matches(eligible_raw + possibly_raw, language)
    rejected_localized = localize_scheme_matches(rejected_raw, language)
    eligible_localized = [x for x in visible_localized if x["status"] == "eligible"]

    all_en_names = ", ".join([f"{i+1}. {x['scheme_name']}" for i, x in enumerate(eligible_raw)]) if eligible_raw else "None"
    all_loc_names = ", ".join([f"{i+1}. {x['scheme_name']}" for i, x in enumerate(eligible_localized)]) if eligible_localized else "None"

    en_reply = (
        f"{confirm_en}All 7 profile steps complete! Based on official scheme rules, you are eligible for {len(eligible_raw)} scheme(s): "
        f"{all_en_names}. "
        f"Check the screen for your required documents checklist, step-by-step application guide, and official myScheme links."
    )

    if language != "en" and language in LOCALIZED_TEMPLATES:
        tmpl = LOCALIZED_TEMPLATES[language]
        loc_reply = f"{confirm_loc}" + tmpl["eligible_summary_all"].format(
            elig=len(eligible_localized),
            all_schemes=all_loc_names
        )
    elif language != "en":
        loc_reply = translate_text(en_reply, "en", language)
    else:
        loc_reply = en_reply

    return {
        "reply": loc_reply,
        "english_reply": en_reply,
        "stage": "results",
        "profile_complete": True,
        "next_field": None,
        "current_step": 7,
        "total_steps": 7,
        "extracted_this_turn": extracted_now,
        "matches": visible_localized,
        "rejected_matches": rejected_localized,
        "rag_chunks": [],
        "csc_guidance": get_csc_info(profile.get("state"), language),
        "disclaimer": disclaimer
    }
