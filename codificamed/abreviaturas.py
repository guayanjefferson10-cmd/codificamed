"""Diccionario de abreviaturas clínicas frecuentes en notas de Ecuador y Latinoamérica.

Las claves se comparan sin distinguir mayúsculas. Si una sigla es ambigua se resuelve por
contexto en `preprocesamiento.expandir_abreviaturas` (ver REGLAS_CONTEXTO).
"""

ABREVIATURAS: dict[str, str] = {
    # Cardiovascular
    "HTA": "hipertensión arterial",
    "IAM": "infarto agudo de miocardio",
    "IAMCEST": "infarto agudo de miocardio con elevación del ST",
    "IAMSEST": "infarto agudo de miocardio sin elevación del ST",
    "SCA": "síndrome coronario agudo",
    "ICC": "insuficiencia cardíaca congestiva",
    "IC": "insuficiencia cardíaca",
    "FA": "fibrilación auricular",
    "RVR": "respuesta ventricular rápida",
    "ACV": "accidente cerebrovascular",
    "ECV": "enfermedad cerebrovascular",
    "AIT": "accidente isquémico transitorio",
    "TEP": "tromboembolia pulmonar",
    "TVP": "trombosis venosa profunda",
    "HSA": "hemorragia subaracnoidea",
    "CI": "cardiopatía isquémica",
    # Endocrino / metabólico
    "DM": "diabetes mellitus",
    "DM1": "diabetes mellitus tipo 1",
    "DM2": "diabetes mellitus tipo 2",
    "DMT2": "diabetes mellitus tipo 2",
    "DMNID": "diabetes mellitus no insulinodependiente",
    "DMID": "diabetes mellitus insulinodependiente",
    "CAD": "cetoacidosis diabética",
    "IMC": "índice de masa corporal",
    # Respiratorio
    "EPOC": "enfermedad pulmonar obstructiva crónica",
    "NAC": "neumonía adquirida en la comunidad",
    "IRA": "infección respiratoria aguda",
    "IRAG": "infección respiratoria aguda grave",
    "SDRA": "síndrome de dificultad respiratoria aguda",
    "SAOS": "síndrome de apnea obstructiva del sueño",
    "TB": "tuberculosis",
    "TBC": "tuberculosis",
    "BK": "baciloscopía",
    "BK+": "baciloscopía positiva",
    # Renal / urológico
    "ERC": "enfermedad renal crónica",
    "IRC": "insuficiencia renal crónica",
    "LRA": "lesión renal aguda",
    "ITU": "infección del tracto urinario",
    "IVU": "infección de vías urinarias",
    "HPB": "hiperplasia prostática benigna",
    "EMO": "examen elemental y microscópico de orina",
    # Digestivo
    "ERGE": "enfermedad por reflujo gastroesofágico",
    "RGE": "reflujo gastroesofágico",
    "EDA": "enfermedad diarreica aguda",
    "HDA": "hemorragia digestiva alta",
    "HDB": "hemorragia digestiva baja",
    "FID": "fosa ilíaca derecha",
    "FII": "fosa ilíaca izquierda",
    "HCD": "hipocondrio derecho",
    "abd": "abdominal",
    # Infeccioso
    "VIH": "virus de la inmunodeficiencia humana",
    "SIDA": "síndrome de inmunodeficiencia adquirida",
    "TARV": "terapia antirretroviral",
    "ITS": "infección de transmisión sexual",
    # Neuro / psiquiátrico / trauma
    "TEC": "traumatismo encefalocraneano",
    "TCE": "traumatismo craneoencefálico",
    "TAG": "trastorno de ansiedad generalizada",
    "TDAH": "trastorno por déficit de atención e hiperactividad",
    # Oncología
    "CA": "cáncer",
    # Exámenes y signos (útiles para no confundir al recuperador)
    "EKG": "electrocardiograma",
    "ECG": "electrocardiograma",
    "Rx": "radiografía",
    "Eco": "ecografía",
    "TAC": "tomografía",
    "RMN": "resonancia magnética",
    "BHCG": "gonadotropina coriónica beta",
    "PCR": "reacción en cadena de la polimerasa",
    "TSH": "hormona estimulante de tiroides",
    "PSA": "antígeno prostático específico",
    "FR": "factor reumatoide",
    "PA": "presión arterial",
    "FC": "frecuencia cardíaca",
    "SatO2": "saturación de oxígeno",
    "MMII": "miembros inferiores",
    "MMSS": "miembros superiores",
    # Redacción clínica
    "Dx": "diagnóstico",
    "IDx": "impresión diagnóstica",
    "Hx": "historia",
    "tto": "tratamiento",
    "Tx": "tratamiento",
    "pcte": "paciente",
    "pte": "paciente",
    "antec": "antecedente",
    "izq": "izquierdo",
    "der": "derecho",
    "G2P1": "gesta 2 para 1",
}

# Siglas cuyo significado depende de lo que sigue.
# (sigla, patrón regex que debe seguir, expansión)
REGLAS_CONTEXTO: list[tuple[str, str, str]] = [
    ("EDA", r"\s*:", "endoscopía digestiva alta"),  # "EDA: úlcera..." es un examen
]
