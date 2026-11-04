from codificamed.catalog import normalizar_codigo
from codificamed.evaluacion import emparejar, evaluar, prf


def test_normalizar():
    assert normalizar_codigo("e119") == "E11.9"
    assert normalizar_codigo("s72.001a") == "S72.0"
    assert normalizar_codigo("I10") == "I10"


def test_emparejar_alternativas_y_opcionales():
    tp, fp, fn, _ = emparejar(["I10", "J44.0|J44.1"], ["J44.1", "I10", "Z87.8", "R05"], ["Z87.8"])
    assert (tp, fp, fn) == (2, 1, 0)


def test_nivel_categoria():
    tp, fp, fn, _ = emparejar(["E11.9"], ["E11.8"], nivel="categoria")
    assert (tp, fp, fn) == (1, 0, 0)
    tp, fp, fn, _ = emparejar(["E11.9"], ["E11.8"], nivel="exacto")
    assert (tp, fp, fn) == (0, 1, 1)


def test_prf():
    assert prf(2, 1, 1)["f1"] == round(2 * (2 / 3) * (2 / 3) / (4 / 3), 4)


def test_evaluar_negados():
    casos = [{"id": "x", "codigos": ["I10"], "negados": ["R50.9"]}]
    preds = [{"codigos": [{"codigo": "I10", "estado": "confirmado", "inicio": 0},
                          {"codigo": "R50.9", "estado": "confirmado", "inicio": 5}],
              "latencia_total_s": 1.0}]
    m = evaluar(casos, preds)
    assert m["negacion"]["confirmadas_por_error"] == 1
    assert m["exacto"]["tp"] == 1 and m["exacto"]["fp"] == 1
