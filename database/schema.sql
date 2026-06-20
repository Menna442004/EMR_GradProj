 CREATE TABLE IF NOT EXISTS doctors (
        doctor_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name     TEXT,
        specialty     TEXT,
        created_at    TEXT
    );

    CREATE TABLE IF NOT EXISTS patients (
        patient_id        TEXT PRIMARY KEY,
        full_name         TEXT,
        date_of_birth     TEXT,
        age               INTEGER,
        gender            TEXT,
        blood_type        TEXT,
        phone             TEXT,
        email             TEXT,
        insurance_status  TEXT,
        emergency_contact TEXT,
        allergies         TEXT,
        created_at        TEXT,
        username          TEXT UNIQUE,
        password_hash     TEXT
    );

    CREATE TABLE IF NOT EXISTS visits (
        visit_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id          TEXT REFERENCES patients(patient_id),
        doctor_id           INTEGER REFERENCES doctors(doctor_id),
        visit_datetime      TEXT,
        visit_type          TEXT,
        department          TEXT,
        chief_complaint     TEXT,
        asr_transcription   TEXT,
        symptom_top1        TEXT,
        symptom_top1_score  REAL,
        symptom_top2        TEXT,
        symptom_top2_score  REAL,
        symptom_top3        TEXT,
        symptom_top3_score  REAL,
        urgency             TEXT,
        severity_score      INTEGER,
        follow_up_needed    INTEGER,
        doctor_notes        TEXT,
        recommended_steps   TEXT,
        audio_duration_sec  REAL,
        processed_at        TEXT
    );

    CREATE TABLE IF NOT EXISTS diagnoses (
        diagnosis_id INTEGER PRIMARY KEY AUTOINCREMENT,
        visit_id     INTEGER REFERENCES visits(visit_id),
        rank         INTEGER,
        disease      TEXT,
        likelihood   TEXT,
        reasoning    TEXT
    );

    CREATE TABLE IF NOT EXISTS medical_records (
        record_id     INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id    TEXT REFERENCES patients(patient_id),
        doctor_id     INTEGER REFERENCES doctors(doctor_id),
        image_path    TEXT,
        view_position TEXT,
        all_findings  TEXT,
        doctor_notes  TEXT,
        timestamp     TEXT
    );

    CREATE TABLE IF NOT EXISTS chat_messages (
        msg_id       INTEGER PRIMARY KEY AUTOINCREMENT,
        sender_id    INTEGER,
        receiver_id  INTEGER,
        message_body TEXT,
        timestamp    TEXT DEFAULT (datetime('now','localtime'))
    );

    CREATE TABLE IF NOT EXISTS prescriptions (
        prescription_id INTEGER PRIMARY KEY AUTOINCREMENT,
        visit_id        INTEGER REFERENCES visits(visit_id),
        patient_id      TEXT REFERENCES patients(patient_id),
        doctor_id       INTEGER REFERENCES doctors(doctor_id),
        medication_name TEXT NOT NULL,
        dosage          TEXT,
        frequency       TEXT,
        duration        TEXT,
        notes           TEXT,
        issued_at       TEXT
    );

    CREATE TABLE IF NOT EXISTS appointments (
        appointment_id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id     TEXT REFERENCES patients(patient_id),
        doctor_id      INTEGER REFERENCES doctors(doctor_id),
        requested_date TEXT,
        requested_time TEXT,
        reason         TEXT,
        status         TEXT DEFAULT 'pending',
        doctor_notes   TEXT,
        created_at     TEXT
