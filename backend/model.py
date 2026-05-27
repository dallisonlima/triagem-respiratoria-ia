import os
import pandas as pd
import numpy as np
import requests
import joblib
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), "modelo_treinado.joblib")

# Ordem canônica das features — usada no treino E na predição (via api.py).
# Qualquer alteração aqui deve ser refletida em api.py e vice-versa.
FEATURE_KEYS = ['FEBRE', 'TOSSE', 'GARGANTA', 'DISPNEIA', 'ASMA', 'DIABETES', 'CARDIOPATI', 'SATURACAO']

_modelo_treinado = None
_accuracy        = 0.0
_samples         = 0

# Tenta carregar modelo persistido ao iniciar o processo
try:
    if os.path.exists(MODEL_PATH):
        dados_salvos     = joblib.load(MODEL_PATH)
        _modelo_treinado = dados_salvos["modelo"]
        _accuracy        = dados_salvos["accuracy"]
        _samples         = dados_salvos["samples"]
        print(f"Modelo carregado: acurácia={_accuracy:.1f}% | amostras={_samples:,}")
except Exception as e:
    print(f"Aviso: Não foi possível carregar o modelo salvo: {e}")

load_dotenv()


# ──────────────────────────────────────────────────────────────
# Coleta de dados e treinamento
# ──────────────────────────────────────────────────────────────

def obter_dados_e_treinar():
    global _modelo_treinado, _accuracy, _samples

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise ValueError(
            "Configurações do Supabase (SUPABASE_URL ou SUPABASE_KEY) "
            "não encontradas no arquivo .env."
        )

    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }

    # Paginação: busca até 50 000 registros em blocos de 1 000
    all_data  = []
    limit     = 1000
    max_total = 50_000

    for offset in range(0, max_total, limit):
        endpoint = f"{url}/rest/v1/srag?select=*&limit={limit}&offset={offset}"
        response = requests.get(endpoint, headers=headers, timeout=10)

        if not response.ok:
            raise ValueError(f"Erro ao buscar do Supabase: {response.text}")

        data = response.json()
        if not data:
            break

        all_data.extend(data)
        if len(data) < limit:
            break

    df = pd.DataFrame(all_data)

    if len(df) < 2:
        raise ValueError("Tabela vazia ou com dados insuficientes no Supabase.")

    df.columns = df.columns.str.strip().str.upper()
    df = df.replace('"', '', regex=True)

    X, Y = _extrair_features(df)

    # ── Busca os feedbacks (retroalimentação local) ─────────────
    X_fb, Y_fb = [], []
    try:
        endpoint_fb = f"{url}/rest/v1/feedbacks?select=desfecho_real_grave,historico(*)"
        resp_fb = requests.get(endpoint_fb, headers=headers, timeout=10)
        if resp_fb.ok:
            fb_data = resp_fb.json()
            X_fb, Y_fb = _extrair_features_feedbacks(fb_data)
            print(f"Buscados {len(X_fb)} registros de feedback validados.")
    except Exception as e:
        print(f"Erro ao buscar feedbacks: {e}")

    # Junta os dados do SIVEP-Gripe originais com os feedbacks
    X.extend(X_fb)
    Y.extend(Y_fb)

    if len(X) < 20:
        raise ValueError("Foram encontradas poucas linhas válidas de pacientes.")

    X = np.array(X)
    Y = np.array(Y)

    # ── Divisão treino / teste (80/20) ──────────────────────────
    # Garante avaliação honesta do modelo em dados não vistos.
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    # ── Regressão Logística com balanceamento de classes ────────
    # class_weight='balanced' compensa a desproporção entre casos
    # leves (maioria) e graves (minoria) nos dados do SIVEP-Gripe.
    clf = LogisticRegression(
        random_state=42,
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
    )
    clf.fit(X_train, Y_train)

    # Acurácia avaliada no conjunto de TESTE (não visto no treino)
    Y_pred   = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, Y_pred) * 100.0

    _accuracy        = accuracy
    _samples         = len(X)        # total de amostras válidas processadas
    _modelo_treinado = clf

    print(
        f"Treino concluído | amostras totais={_samples:,} "
        f"| acurácia (teste)={_accuracy:.1f}%"
    )

    # Persiste o modelo para evitar retreinamento a cada restart
    try:
        joblib.dump(
            {"modelo": _modelo_treinado, "accuracy": _accuracy, "samples": _samples},
            MODEL_PATH,
        )
    except Exception as e:
        print(f"Aviso: Falha ao salvar o modelo: {e}")

    return _accuracy, _samples


# ──────────────────────────────────────────────────────────────
# Extração de features (lógica isolada para facilitar testes)
# ──────────────────────────────────────────────────────────────

def _extrair_features(df: pd.DataFrame):
    """
    Percorre o DataFrame e retorna (X, Y) prontos para o sklearn.
    Segue a ordem definida em FEATURE_KEYS.
    """
    X, Y = [], []

    for _, row in df.iterrows():
        is_grave       = False
        has_valid_target = False

        # ── Lógica de gravidade (UTI / Evolução / Suporte ventilatório) ──
        if 'UTI' in row:
            try:
                uti = int(float(str(row['UTI'])))
                if uti in [1, 2]:
                    has_valid_target = True
                if uti == 1:
                    is_grave = True
            except Exception:
                pass

        if 'EVOLUCAO' in row:
            try:
                evo = int(float(str(row['EVOLUCAO'])))
                if evo in [1, 2]:
                    has_valid_target = True
                if evo == 2:          # Óbito
                    is_grave = True
            except Exception:
                pass

        if 'SUPORT_VEN' in row:
            try:
                sup = int(float(str(row['SUPORT_VEN'])))
                if sup in [1, 2, 3]:
                    has_valid_target = True
                if sup in [1, 2]:     # Suporte ventilatório invasivo/não-invasivo
                    is_grave = True
            except Exception:
                pass

        if not has_valid_target:
            continue

        # ── Idade em anos ────────────────────────────────────────
        idade_anos = 0.0
        try:
            if 'NU_IDADE_N' in row and 'TP_IDADE' in row:
                tp_idade   = int(float(row['TP_IDADE']))
                nu_idade   = int(float(row['NU_IDADE_N']))
                if tp_idade == 3:
                    idade_anos = float(nu_idade)
                elif tp_idade == 2:
                    idade_anos = nu_idade / 12.0
        except Exception:
            pass

        if np.isnan(idade_anos) or idade_anos > 120:
            continue

        # ── Features binárias (ordem = FEATURE_KEYS) ────────────
        # Nota: CARDIOPATI está sem o "A" final — o DataSUS limita
        # nomes de colunas a 10 caracteres no formato DBF.
        row_features = []
        row_valid    = True

        for feat in FEATURE_KEYS:
            val = 2  # padrão DataSUS: 2 = Não
            try:
                if feat in row:
                    val = int(float(str(row[feat])))
            except Exception:
                pass

            # Valor 9 = ignorado no padrão DataSUS → descarta a linha
            if val not in [1, 2]:
                row_valid = False
                break

            row_features.append(1 if val == 1 else 0)

        if row_valid:
            row_features.append(idade_anos / 100.0)   # normalização
            X.append(row_features)
            Y.append(1 if is_grave else 0)

    return X, Y


def _extrair_features_feedbacks(feedbacks_list: list):
    """
    Extrai as features e targets dos registros de feedback validados pelo médico.
    Mapeia os dados do histórico (booleanos) para o formato do classificador (0 e 1).
    """
    X_fb, Y_fb = [], []
    
    mapping = {
        'FEBRE': 'febre',
        'TOSSE': 'tosse',
        'GARGANTA': 'garganta',
        'DISPNEIA': 'dispneia',
        'ASMA': 'asma',
        'DIABETES': 'diabetes',
        'CARDIOPATI': 'cardiopatia',
        'SATURACAO': 'saturacao'
    }

    for item in feedbacks_list:
        try:
            is_grave = item.get("desfecho_real_grave", False)
            hist = item.get("historico")
            
            # Se a FK estiver vazia por algum motivo
            if not hist or not isinstance(hist, dict):
                continue
                
            idade = hist.get("idade", 0)
            if idade > 120:
                continue

            row_features = []
            for key in FEATURE_KEYS:
                val_bool = hist.get(mapping[key], False)
                row_features.append(1 if val_bool else 0)
            
            row_features.append(idade / 100.0)
            
            X_fb.append(row_features)
            Y_fb.append(1 if is_grave else 0)
        except Exception:
            pass

    return X_fb, Y_fb


# ──────────────────────────────────────────────────────────────
# Predição e status
# ──────────────────────────────────────────────────────────────

def prever_gravidade(features_list: list) -> float:
    """
    Recebe uma lista de features na mesma ordem de FEATURE_KEYS + idade_norm
    e retorna a probabilidade de quadro grave em percentual (0.0 – 100.0).
    """
    global _modelo_treinado
    if _modelo_treinado is None:
        raise Exception("O modelo de IA ainda não foi treinado.")

    proba = _modelo_treinado.predict_proba([features_list])[0]
    return float(proba[1] * 100.0)


def get_stats() -> dict:
    return {
        "treinado": _modelo_treinado is not None,
        "accuracy": _accuracy,
        "samples":  _samples,
    }