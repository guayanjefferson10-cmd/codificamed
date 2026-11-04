from codificamed.negacion import AFIRMADO, NEGADO, SOSPECHA, analizar, estado_de_span
from codificamed.preprocesamiento import expandir_abreviaturas, preprocesar, segmentar_oraciones


def estado(frase, termino):
    toks, _ = analizar(frase)
    i = frase.index(termino)
    return estado_de_span(toks, i, i + len(termino))


def test_negacion_basica():
    assert estado("Niega fiebre", "fiebre") == NEGADO
    assert estado("Se descarta neumonía", "neumonía") == NEGADO
    assert estado("Se decarta artritis reumatoide", "artritis") == NEGADO   # con error de tipeo
    assert estado("Sin signos de sangrado", "sangrado") == NEGADO
    assert estado("Test de embarazo negativo", "embarazo") == NEGADO        # negación posterior


def test_afirmados_y_falsos_positivos():
    assert estado("DM2 sin complicaciones", "DM2") == AFIRMADO
    assert estado("Apéndice no compresible", "Apéndice") == AFIRMADO
    assert estado("Neumonía, no especificada", "Neumonía") == AFIRMADO
    assert estado("Dengue sin signos de alarma", "Dengue") == AFIRMADO
    assert estado("Niega fiebre pero refiere tos", "tos") == AFIRMADO       # terminador


def test_sospecha():
    assert estado("Probable neumonía", "neumonía") == SOSPECHA
    assert estado("A descartar apendicitis", "apendicitis") == SOSPECHA


def test_abreviaturas_y_contexto():
    assert "hipertensión arterial" in expandir_abreviaturas("HTA en tto")
    assert expandir_abreviaturas("EDA: úlcera").startswith("endoscopía digestiva alta")
    assert "enfermedad diarreica aguda" in expandir_abreviaturas("EDA por rotavirus")


def test_segmentacion_respeta_decimales():
    t = "Fiebre de 38.2 grados. Plaquetas 120.000. H. pylori positivo"
    assert len(segmentar_oraciones(t)) == 3


def test_correccion_ortografica():
    n = preprocesar("Dx de gonartosis bilateral")
    assert "gonartrosis" in n.clausulas[0].consulta


def test_offsets_conservados():
    nota = "Pcte con HTA. Niega fiebre."
    n = preprocesar(nota)
    for c in n.clausulas:
        assert nota[c.inicio:c.fin] == c.texto
