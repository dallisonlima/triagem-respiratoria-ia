import os
import requests
import streamlit as st
from datetime import date, timedelta
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

load_dotenv()

# ── Ambiente ──────────────────────────────────────────────────
def _secret(key: str, default: str) -> str:
    if val := os.environ.get(key):
        return val
    try:
        return st.secrets[key]
    except Exception:
        return default

API_BASE      = _secret("API_BASE",      "http://localhost:8000")
TRAIN_SECRET  = _secret("TRAIN_SECRET",  "")
SUPABASE_URL  = _secret("SUPABASE_URL",  "")
SUPABASE_KEY  = _secret("SUPABASE_KEY",  "")

AUTH_HEADERS  = {
    "apikey":        SUPABASE_KEY,
    "Content-Type":  "application/json",
}

# ── Config da página ──────────────────────────────────────────
st.set_page_config(
    page_title="MedTriagem IA",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# ── Design system externo ─────────────────────────────────────
def _load_css(path: str):
    with open(path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css(os.path.join(os.path.dirname(__file__), "style.css"))

# ══════════════════════════════════════════════════════════════
# RECONSTRUTOR DE SESSÃO ANTES DAS TELAS (CRÍTICO)
# ══════════════════════════════════════════════════════════════
if "user" not in st.session_state:
    st.session_state.user = None

# Tenta recuperar imediatamente pelo cookie para não deslogar ao mudar a URL
if st.session_state.user is None:
    cookie_val = st.context.cookies.get("medtriagem_user")
    if cookie_val:
        try:
            import urllib.parse
            import json
            decoded = urllib.parse.unquote(cookie_val)
            st.session_state.user = json.loads(decoded)
        except Exception:
            pass

# Inicializa outros estados padrões se não existirem
defaults = {
    "step":         "train",
    "trainedModel": None,
    "resultado":    None,
}
for key, default in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default


def validar_cpf(cpf_str: str) -> bool:
    """Valida se o CPF informado é válido."""
    cpf = "".join(filter(str.isdigit, cpf_str))
    if len(cpf) != 11:
        return False
    if cpf == cpf[0] * 11:
        return False
    soma = sum(int(cpf[i]) * (10 - i) for i in range(9))
    resto = soma % 11
    digito1 = 0 if resto < 2 else 11 - resto
    if int(cpf[9]) != digito1:
        return False
    soma = sum(int(cpf[i]) * (11 - i) for i in range(10))
    resto = soma % 11
    digito2 = 0 if resto < 2 else 11 - resto
    return int(cpf[10]) == digito2

# ══════════════════════════════════════════════════════════════
# AUTENTICAÇÃO — Supabase Auth REST API
# ══════════════════════════════════════════════════════════════

def _set_user_cookie(user_data: dict):
    """Grava o cookie no navegador de forma persistente."""
    import urllib.parse
    import json
    
    val = urllib.parse.quote(json.dumps(user_data))
    js_code = f"""
    <script>
        var date = new Date();
        date.setTime(date.getTime() + (7 * 24 * 60 * 60 * 1000));
        var expires = "; expires=" + date.toUTCString();
        document.cookie = "medtriagem_user=" + "{val}" + expires + "; path=/; SameSite=Lax";
    </script>
    """
    st.html(js_code)
    st.rerun()


def _delete_user_cookie():
    """Remove o cookie do navegador e limpa estados locais."""
    js_code = """
    <script>
        document.cookie = "medtriagem_user=; expires=Thu, 01 Jan 1970 00:00:00 UTC; path=/; SameSite=Lax";
    </script>
    """
    st.html(js_code)
    st.session_state.clear()
    st.rerun()

def _supabase_login(email: str, password: str) -> dict:
    """Autentica via POST /auth/v1/token?grant_type=password."""
    resp = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
        headers=AUTH_HEADERS,
        json={"email": email, "password": password},
        timeout=10,
    )
    return resp.json(), resp.status_code

def _supabase_signup(email: str, password: str) -> dict:
    """Cadastra via POST /auth/v1/signup."""
    resp = requests.post(
        f"{SUPABASE_URL}/auth/v1/signup",
        headers=AUTH_HEADERS,
        json={"email": email, "password": password},
        timeout=10,
    )
    return resp.json(), resp.status_code

def _error_msg(data: dict) -> str:
    """Extrai a mensagem de erro da resposta do Supabase e traduz."""
    raw = data.get("error_description") or data.get("msg") or data.get("message", "")
    translations = {
        "Invalid login credentials":         "Email ou senha incorretos.",
        "User already registered":           "Este email já está cadastrado.",
        "Password should be at least 6 characters":
            "A senha deve ter no mínimo 6 caracteres.",
        "Unable to validate email address: invalid format":
            "Formato de email inválido.",
    }
    return translations.get(raw, raw or "Erro desconhecido.")

def do_logout():
    """Limpa toda a sessão e volta para o login."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    _delete_user_cookie()
    st.stop()

# ══════════════════════════════════════════════════════════════
# PROCESSAMENTO PREVENTIVO DE FEEDBACKS (ANTES DA TELA DE LOGIN)
# ══════════════════════════════════════════════════════════════
params = st.query_params

if "fb_id" in params and "fb_grave" in params and st.session_state.get("user") is not None:
    try:
        f_id = int(params["fb_id"])
        f_grave = params["fb_grave"] == "1"
        
        token = st.session_state.user.get("token", "")
        headers = {
            "apikey":        SUPABASE_KEY,
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
            "Prefer":        "return=minimal",
        }
        payload = {
            "id_historico": f_id,
            "desfecho_real_grave": f_grave
        }
        
        resp = requests.post(
            f"{SUPABASE_URL}/rest/v1/feedbacks",
            headers=headers,
            json=payload,
            timeout=10,
        )
        
        if resp.ok:
            st.toast("✅ Evolução confirmada com sucesso!", icon="🩺")
            
    except Exception:
        pass
    finally:
        # Remove os parâmetros de forma limpa sem estragar o estado ou cookies
        chaves_para_remover = [k for k in ["fb_id", "fb_grave"] if k in st.query_params]
        for k in chaves_para_remover:
            del st.query_params[k]
        
        # Garante a permanência do médico na mesma tela
        st.session_state.step = "historico"
        st.rerun()

# ══════════════════════════════════════════════════════════════
# TELA INTERRUPTIVA: LOGIN / CADASTRO
# ══════════════════════════════════════════════════════════════

if st.session_state.user is None:

    st.markdown("""
    <div class="login-wrapper">
        <div class="login-icon">🩺</div>
        <div class="login-title">MedTriagem IA</div>
        <div class="login-subtitle">Triagem Respiratória · SIVEP-Gripe · DataSUS</div>
    </div>
    """, unsafe_allow_html=True)

    tab_login, tab_signup = st.tabs(["🔑 Entrar", "📝 Criar Conta"])

    with tab_login:
        with st.form("login_form"):
            email_login = st.text_input("Email", placeholder="seu@email.com", key="login_email")
            senha_login = st.text_input("Senha", type="password", placeholder="••••••••", key="login_senha")
            submit_login = st.form_submit_button("Entrar", type="primary", use_container_width=True)

        if submit_login:
            if not email_login or not senha_login:
                st.error("Preencha email e senha.")
            else:
                with st.spinner("Autenticando..."):
                    data, status = _supabase_login(email_login, senha_login)
                if status == 200 and "access_token" in data:
                    user_email = data.get("user", {}).get("email", email_login)
                    st.session_state.user = {
                        "email": user_email,
                        "token": data["access_token"],
                    }
                    _set_user_cookie(st.session_state.user)
                    
                    # Rota limpa manual: vai direto para a tela inicial configurada
                    st.session_state.step = "train"
                    st.rerun()
                else:
                    st.error(_error_msg(data))

    with tab_signup:
        with st.form("signup_form"):
            email_signup  = st.text_input("Email", placeholder="seu@email.com", key="signup_email")
            senha_signup  = st.text_input("Senha", type="password", placeholder="Mínimo 6 caracteres", key="signup_senha")
            senha_confirm = st.text_input("Confirmar Senha", type="password", placeholder="Repita a senha", key="signup_confirm")
            submit_signup = st.form_submit_button("Criar Conta", type="primary", use_container_width=True)

        if submit_signup:
            if not email_signup or not senha_signup:
                st.error("Preencha todos os campos.")
            elif senha_signup != senha_confirm:
                st.error("As senhas não coincidem.")
            elif len(senha_signup) < 6:
                st.error("A senha deve ter no mínimo 6 caracteres.")
            else:
                with st.spinner("Criando conta..."):
                    data, status = _supabase_signup(email_signup, senha_signup)
                if status in (200, 201) and data.get("id"):
                    st.success("✅ Conta criada com sucesso! Faça login na aba **Entrar**.")
                elif status in (200, 201) and data.get("access_token"):
                    user_email = data.get("user", {}).get("email", email_signup)
                    st.session_state.user = {
                        "email": user_email,
                        "token": data["access_token"],
                    }
                    _set_user_cookie(st.session_state.user)
                else:
                    st.error(_error_msg(data))

    st.markdown("""
    <div class="med-caption">
        ⚕️ Ferramenta de apoio clínico — não substitui avaliação médica presencial.<br>
        Dados: SIVEP-Gripe (DataSUS) · Modelo: Regressão Logística (Scikit-Learn) · Backend: FastAPI
    </div>
    """, unsafe_allow_html=True)

    st.stop()

# ══════════════════════════════════════════════════════════════
# ÁREA AUTENTICADA (Executada apenas se houver Usuário Ativo)
# ══════════════════════════════════════════════════════════════

user_email = st.session_state.user["email"]

with st.container(key="topbar"):
    col_logo, col_email, col_sair = st.columns([5, 3, 1])
    with col_logo:
        st.markdown(
            '<div class="topbar-logo">'
            '<span class="topbar-icon">🩺</span>'
            '<div>'
            '<p class="topbar-title">MedTriagem IA</p>'
            '<p class="topbar-sub">Triagem Respiratória · SIVEP-Gripe · DataSUS</p>'
            '</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    with col_email:
        st.markdown(f'<div class="topbar-user">👤 {user_email}</div>', unsafe_allow_html=True)
    with col_sair:
        if st.button("Sair", key="btn_logout"):
            do_logout()

if st.session_state.trainedModel:
    m = st.session_state.trainedModel
    st.markdown(
        f'<div class="status-badge">'
        f'<span class="status-dot"></span>'
        f'IA ATIVA &nbsp;·&nbsp; {m["samples"]:,} pacientes &nbsp;·&nbsp; acurácia {m["accuracy"]:.1f}%'
        f'</div>',
        unsafe_allow_html=True,
    )

# ── Funções auxiliares ────────────────────────────────────────
@st.cache_data(ttl=10800, show_spinner=False)
def fetch_and_train(secret: str):
    headers = {"X-Train-Secret": secret} if secret else {}
    r = requests.post(f"{API_BASE}/train", headers=headers, timeout=300)
    r.raise_for_status()
    return r.json()

def handle_train():
    try:
        with st.spinner("Conectando ao backend e treinando o modelo..."):
            data = fetch_and_train(TRAIN_SECRET)
            st.session_state.trainedModel = {
                "accuracy": data.get("accuracy"),
                "samples":  data.get("samples"),
            }
            st.session_state.step = "form"
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", str(e))
        except Exception:
            detail = str(e)
        st.error(f"Erro do Backend: {detail}")
        st.cache_data.clear()
    except Exception as e:
        st.error(f"Erro de conexão: {e}")
        st.cache_data.clear()


def _label_sintoma(key: str) -> str:
    return {
        "febre":       "🌡️ Febre",
        "tosse":       "😮‍💨 Tosse",
        "dispneia":    "💨 Falta de Ar",
        "garganta":    "🔴 Dor de Garganta",
        "saturacao":   "🩸 Sat. O₂ < 95%",
        "asma":        "🌬️ Asma",
        "diabetes":    "💉 Diabetes",
        "cardiopatia": "❤️ Cardiopatia",
    }.get(key, key)


def render_footer():
    st.markdown("""
    <div class="med-caption">
        ⚕️ Ferramenta de apoio clínico — não substitui avaliação médica presencial.<br>
        Dados: SIVEP-Gripe (DataSUS) · Modelo: Regressão Logística (Scikit-Learn) · Backend: FastAPI
    </div>
    """, unsafe_allow_html=True)

# ── Histórico — Supabase ──────────────────────────────────────
def salvar_historico(form_data: dict, prob: float, classificacao: str):
    usuario = st.session_state.user.get("email", "Usuário")
    token   = st.session_state.user.get("token", "")
    
    payload_full = {
        "usuario":       usuario,
        "idade":         form_data["idade"],
        "nome":          form_data.get("nome", ""),
        "sexo":          form_data.get("sexo", ""),
        "cpf":           form_data.get("cpf", ""),
        "febre":         form_data["febre"],
        "tosse":         form_data["tosse"],
        "dispneia":      form_data["dispneia"],
        "garganta":      form_data["garganta"],
        "saturacao":     form_data["saturacao"],
        "asma":          form_data["asma"],
        "diabetes":      form_data["diabetes"],
        "cardiopatia":   form_data["cardiopatia"],
        "probabilidade": round(prob, 2),
        "classificacao": classificacao,
    }
    
    payload_compat = {
        "usuario":       usuario,
        "idade":         form_data["idade"],
        "febre":         form_data["febre"],
        "tosse":         form_data["tosse"],
        "dispneia":      form_data["dispneia"],
        "garganta":      form_data["garganta"],
        "saturacao":     form_data["saturacao"],
        "asma":          form_data["asma"],
        "diabetes":      form_data["diabetes"],
        "cardiopatia":   form_data["cardiopatia"],
        "probabilidade": round(prob, 2),
        "classificacao": classificacao,
    }
    
    headers = {
        "apikey":        SUPABASE_KEY,
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
        "Prefer":        "return=minimal",
    }
    
    try:
        resp = requests.post(f"{SUPABASE_URL}/rest/v1/historico", headers=headers, json=payload_full, timeout=10)
        if resp.status_code == 400:
            requests.post(f"{SUPABASE_URL}/rest/v1/historico", headers=headers, json=payload_compat, timeout=10)
    except Exception:
        pass

def carregar_historico() -> list:
    usuario = st.session_state.user.get("email", "")
    token   = st.session_state.user.get("token", "")
    try:
        resp = requests.get(
            f"{SUPABASE_URL}/rest/v1/historico"
            f"?select=*,feedbacks(desfecho_real_grave)&usuario=eq.{usuario}&order=criado_em.desc&limit=50",
            headers={
                "apikey":        SUPABASE_KEY,
                "Authorization": f"Bearer {token}",
            },
            timeout=10,
        )
        if resp.ok:
            return resp.json()
    except Exception:
        pass
    return []

# ══════════════════════════════════════════════════════════════
# RENDERIZAÇÃO DAS TELAS INTERNAS
# ══════════════════════════════════════════════════════════════

# ── TELA: TREINAR ─────────────────────────────────────────────
if st.session_state.step == "train":
    st.markdown("""
    <div class="hero-section">
        <p class="hero-eyebrow">Sistema de Diagnóstico Respiratório</p>
        <h1 class="hero-title">Triagem Clínica<br><span>com Inteligência Artificial</span></h1>
        <p class="hero-desc">
            Acesse o sistema de diagnóstico respiratório treinado com dados reais do SIVEP-Gripe (DataSUS).
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("⚡  Iniciar Treinamento da IA →", type="primary"):
        handle_train()
        st.rerun()

    st.markdown("""
    <div class="metrics-row">
        <div class="metric-card">
            <div class="metric-card-label">Algoritmo</div>
            <div class="metric-card-value metric-card-accent">Reg. Logística</div>
        </div>
        <div class="metric-card">
            <div class="metric-card-label">Dataset</div>
            <div class="metric-card-value">SIVEP-Gripe</div>
        </div>
        <div class="metric-card">
            <div class="metric-card-label">Backend</div>
            <div class="metric-card-value">FastAPI</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    render_footer()
    
# ── TELA: FORMULÁRIO ──────────────────────────────────────────
elif st.session_state.step == "form":
    st.markdown("""
    <div class="form-title">Avaliação do Paciente</div>
    <div class="form-subtitle">Preencha os dados clínicos para análise pela IA</div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="form-section-title">Dados Básicos</div>', unsafe_allow_html=True)
    nome = st.text_input("Nome Completo do Paciente", placeholder="Ex: Maria da Silva", key="paciente_nome")
    
    c1, c2, c3 = st.columns([2, 2, 0.8])
    with c1:
        cpf_col = st.container()

    with cpf_col:
        def formatar_cpf_input():
            valor = st.session_state.paciente_cpf
            numeros = "".join(filter(str.isdigit, valor))[:11]
            if len(numeros) <= 3:
                st.session_state.paciente_cpf = numeros
            elif len(numeros) <= 6:
                st.session_state.paciente_cpf = f"{numeros[:3]}.{numeros[3:]}"
            elif len(numeros) <= 9:
                st.session_state.paciente_cpf = f"{numeros[:3]}.{numeros[3:6]}.{numeros[6:]}"
            else:
                st.session_state.paciente_cpf = f"{numeros[:3]}.{numeros[3:6]}.{numeros[6:9]}-{numeros[9:11]}"

        cpf = st.text_input("CPF", placeholder="000.000.000-00", key="paciente_cpf", max_chars=14, on_change=formatar_cpf_input)
        cpf_numeros = "".join(filter(str.isdigit, cpf))
        
        if cpf_numeros:
            if len(cpf_numeros) < 11:
                st.markdown('<div class="cpf-feedback cpf-digitando">⏳ Digitando...</div>', unsafe_allow_html=True)
            elif validar_cpf(cpf):
                st.markdown('<div class="cpf-feedback cpf-valido">✅ CPF válido</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="cpf-feedback cpf-invalido">❌ CPF inválido</div>', unsafe_allow_html=True)

    sexo = st.selectbox("Sexo", options=["Feminino", "Masculino"], index=None, placeholder="Selecione", key="paciente_sexo")
    
    with c2:
        data_minima = date.today() - timedelta(days=130*365)
        data_nasc = st.date_input("Nascimento", value=None, min_value=data_minima, max_value=date.today(), format="DD/MM/YYYY", key="paciente_nascimento")
        if data_nasc:
            hoje = date.today()
            idade = hoje.year - data_nasc.year - ((hoje.month, hoje.day) < (data_nasc.month, data_nasc.day))
        else:
            idade = "—"
            
    with c3:
        st.markdown(f'<div class="idade-badge"><span class="idade-valor">{idade}</span><span class="idade-label">anos</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="form-section-title">Sintomas Principais</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        febre    = st.checkbox("🌡️ Febre")
        tosse    = st.checkbox("😮‍💨 Tosse")
        dispneia = st.checkbox("💨 Falta de Ar / Dispneia")
    with c2:
        garganta  = st.checkbox("🔴 Dor de Garganta")
        saturacao = st.checkbox("🩸 Saturação O₂ < 95%")

    st.markdown('<div class="form-section-title">Comorbidades</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    asma        = c1.checkbox("🌬️ Asma")
    diabetes    = c2.checkbox("💉 Diabetes")
    cardiopatia = c3.checkbox("❤️ Cardiopatia")

    algum_selecionado = any([febre, tosse, dispneia, garganta, saturacao, asma, diabetes, cardiopatia])
    cpf_valido = validar_cpf(cpf) if cpf else False
    dados_preenchidos = bool(nome and nome.strip() and cpf_valido and sexo and data_nasc)
    pode_enviar = algum_selecionado and dados_preenchidos

    if not dados_preenchidos:
        st.markdown('<div class="aviso aviso-erro">⚠️ Preencha todos os Dados Básicos (Nome, CPF válido, Sexo e Nascimento).</div>', unsafe_allow_html=True)
    if not algum_selecionado:
        st.markdown('<div class="aviso aviso-alerta">⚠️ Selecione ao menos um sintoma ou comorbidade para prosseguir.</div>', unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        analisar_clicado = st.button("Analisar com IA →", type="primary", disabled=not pode_enviar, use_container_width=True)
    with col_btn2:
        if st.button("📋 Ver Histórico", use_container_width=True):
            st.session_state.step = "historico"
            st.rerun()

    if analisar_clicado:
        payload_ia = {
            "idade": idade, "febre": febre, "tosse": tosse,
            "garganta": garganta, "dispneia": dispneia, "saturacao": saturacao,
            "asma": asma, "diabetes": diabetes, "cardiopatia": cardiopatia,
        }
        form_data = {"nome": nome, "sexo": sexo, "cpf": cpf, **payload_ia}
        prob_gravidade, classificacao = 0.0, "leve"
        limiares = {"grave": 60.0, "moderado": 30.0}
        erro_backend = None

        try:
            resp = requests.post(f"{API_BASE}/predict", json=payload_ia, timeout=15)
            resp.raise_for_status()
            payload        = resp.json()
            prob_gravidade = payload.get("probabilidadeGravidade", 0.0)
            classificacao  = payload.get("classificacao", "leve")
            limiares       = payload.get("limiares", limiares)
        except Exception as e:
            erro_backend = str(e)

        st.session_state.resultado = {
            "probabilidadeGravidade": prob_gravidade,
            "classificacao":          classificacao,
            "limiares":               limiares,
            "formData":               form_data,
            "erroBackend":            erro_backend,
        }
        if not erro_backend:
            salvar_historico(form_data, prob_gravidade, classificacao)
        st.session_state.step = "result"
        st.rerun()
        
    render_footer()

# ── TELA: RESULTADO ───────────────────────────────────────────
elif st.session_state.step == "result":
    resultado     = st.session_state.resultado
    prob          = resultado["probabilidadeGravidade"]
    classificacao = resultado["classificacao"]
    fd            = resultado["formData"]

    if resultado.get("erroBackend"):
        st.error(f"⚠️ Não foi possível consultar a IA: {resultado['erroBackend']}")

    cfg = {
        "grave":    ("result-grave",    "result-label-grave",    "🚨", "QUADRO GRAVE",    "Risco elevado de internação ou UTI. Encaminhar imediatamente."),
        "moderado": ("result-moderado", "result-label-moderado", "⚠️", "QUADRO MODERADO", "Avaliação médica urgente recomendada nas próximas horas."),
        "leve":     ("result-leve",     "result-label-leve",     "✅", "QUADRO LEVE",     "Monitoramento domiciliar recomendado. Retornar se piorar."),
    }.get(classificacao, ("result-leve", "result-label-leve", "✅", "QUADRO LEVE", ""))

    card_cls, label_cls, icon, label, desc = cfg
    bar_pct = min(prob, 100)

    st.markdown(f"""
    <div class="result-card {card_cls}">
        <div class="result-icon">{icon}</div>
        <div class="result-label {label_cls}">{label}</div>
        <div class="result-title">{desc}</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([2, 3])
    with c1:
        st.markdown(f'<div class="prob-col-left"><div class="prob-value">{prob:.1f}%</div><div class="prob-label">chance de complicação grave</div></div>', unsafe_allow_html=True)
    with c2:
        lim = resultado.get("limiares", {"moderado": 30.0, "grave": 60.0})
        lim_moderado, lim_grave = lim.get("moderado", 30.0), lim.get("grave", 60.0)
        st.markdown(f"""
        <div class="prob-col-right">
            <div class="prob-bar-wrap">
                <div class="prob-bar-fill prob-bar-{classificacao}" style="width:{bar_pct}%"></div>
                <div class="prob-bar-marker" style="left:{lim_moderado}%"></div>
                <div class="prob-bar-marker" style="left:{lim_grave}%"></div>
            </div>
            <div class="prob-bar-ticks">
                <span class="prob-tick" style="left:0;">0%</span>
                <span class="prob-tick" style="left:{lim_moderado}%;transform:translateX(-50%);">{int(lim_moderado)}%</span>
                <span class="prob-tick" style="left:{lim_grave}%;transform:translateX(-50%);">{int(lim_grave)}%</span>
                <span class="prob-tick" style="right:0;">100%</span>
            </div>
            <div class="prob-bar-labels">
                <span style="width:{lim_moderado}%;">Leve</span>
                <span style="width:{lim_grave - lim_moderado}%;">Moderado</span>
                <span style="width:{100 - lim_grave}%;">Grave</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="form-section-title">Perfil do Paciente</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Nome:** {fd.get('nome', '—') or '—'}")
        st.markdown(f"**CPF:** {fd.get('cpf', '—') or '—'}")
    with col2:
        st.markdown(f"**Sexo:** {fd.get('sexo', '—') or '—'}")
        st.markdown(f"**Idade:** {fd['idade']} anos")

    st.markdown('<div class="form-section-title">Sintomas e Comorbidades</div>', unsafe_allow_html=True)
    sintomas_marcados = [_label_sintoma(k) for k in ["febre", "tosse", "dispneia", "garganta", "saturacao", "asma", "diabetes", "cardiopatia"] if fd.get(k)]

    if sintomas_marcados:
        tags_html = "".join(f'<span class="sintoma-tag">{s}</span>' for s in sintomas_marcados)
        st.markdown(f'<div class="tags-wrap">{tags_html}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="sem-sintomas">Nenhum sintoma ou comorbidade informado.</p>', unsafe_allow_html=True)

    st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("← Nova Avaliação", use_container_width=True):
            st.session_state.step, st.session_state.resultado = "form", None
            st.rerun()
    with col_btn2:
        if st.button("📋 Ver Histórico", type="primary", use_container_width=True):
            st.session_state.step = "historico"
            st.rerun()

    render_footer()

# ── TELA: HISTÓRICO ───────────────────────────────────────────
elif st.session_state.step == "historico":
    st.markdown('<div class="form-title">Histórico de Avaliações</div><div class="form-subtitle">Suas últimas 50 avaliações realizadas</div>', unsafe_allow_html=True)
    
    with st.spinner("Carregando histórico..."):
        registros = carregar_historico()

    if not registros:
        st.markdown('<div class="historico-vazio"><div class="historico-vazio-icon">📋</div><div class="historico-vazio-titulo">Nenhuma avaliação encontrada</div></div>', unsafe_allow_html=True)
    else:
        badge_cfg = {
            "grave":     ("🚨", "#dc2626", "rgba(239,68,68,0.08)",   "rgba(239,68,68,0.2)"),
            "moderado": ("⚠️", "#d97706", "rgba(245,158,11,0.08)",  "rgba(245,158,11,0.2)"),
            "leve":     ("✅", "#059669", "rgba(16,185,129,0.08)",  "rgba(16,185,129,0.2)"),
        }
        sintoma_keys = ["febre", "tosse", "dispneia", "garganta", "saturacao", "asma", "diabetes", "cardiopatia"]

        for reg in registros:
            classi = reg.get("classificacao", "leve")
            icon, cor, bg, border = badge_cfg.get(classi, badge_cfg["leve"])
            prob, idade = reg.get("probabilidade", 0), reg.get("idade", "—")
            reg_nome, reg_sexo, reg_cpf = reg.get("nome"), reg.get("sexo"), reg.get("cpf")
            
            perfil_detalhado = f"🎂 {idade} anos"
            if reg_sexo: perfil_detalhado += f" &nbsp;·&nbsp; 🚻 {reg_sexo}"
            if reg_cpf:  perfil_detalhado += f" &nbsp;·&nbsp; 💳 CPF: {reg_cpf}"

            criado_em = reg.get("criado_em", "")
            try:
                from datetime import datetime
                dt = datetime.fromisoformat(criado_em.replace("Z", "+00:00"))
                data_fmt = dt.astimezone().strftime("%d/%m/%Y %H:%M")
            except Exception:
                data_fmt = criado_em[:16] if criado_em else "—"

            sintomas = [_label_sintoma(k) for k in sintoma_keys if reg.get(k)]
            tags_html = "".join(f'<span class="hist-tag">{s}</span>' for s in sintomas) if sintomas else '<span class="hist-sem-sintoma">Nenhum sintoma registrado</span>'
            nome_html = f'<div class="hist-nome">🧑 {reg_nome}</div>' if reg_nome else ""

            id_hist = reg.get("id", "0")
            feedbacks_do_registro = reg.get("feedbacks", [])

            # 🛠️ PASSO 1: Criamos o container nativo com borda
            with st.container(border=True):
                
                # Montamos as informações internas usando as classes do seu style.css
                card_superior_html = f"""
                <div class="hist-card-body">
                    <div class="hist-card-border-indicator" style="border-left: 5px solid {cor};">
                        <div>
                            <div class="hist-badge" style="background:{bg}; border: 1px solid {border}; color:{cor};">{icon} {classi.upper()}</div>
                            {nome_html}
                            <div class="hist-meta">Data e Hora: 🕐 {data_fmt}</div>
                            <div class="hist-perfil">{perfil_detalhado}</div>
                            <div class="hist-tags">{tags_html}</div>
                        </div>
                        <div class="hist-prob">
                            <div class="hist-prob-valor">{prob:.1f}%</div>
                            <div class="hist-prob-label">prob. complicação</div>
                        </div>
                    </div>
                </div>
                """
                st.markdown(card_superior_html, unsafe_allow_html=True)
                
                # 🛠️ PASSO 2: Parte Inferior (Ações de Voto ou Caixa de Validação)
                if len(feedbacks_do_registro) > 0:
                    texto_desfecho = "Grave" if feedbacks_do_registro[0].get("desfecho_real_grave", False) else "Leve/Moderado"
                    
                    # Utilizando a nova classe configurada no CSS
                    st.markdown(f"""
                    <div class="hist-validacao-box">
                        ✅ Validado pelo médico como: {texto_desfecho}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    # Divisória sutil nativa em HTML entre o corpo e as ações
                    st.markdown("""<hr style="margin: 14px 0px; border: 0; border-top: 1px solid rgba(148,163,184,0.15);">""", unsafe_allow_html=True)

                    # Botões nativos aplicados de forma limpa dentro das colunas do container
                    c_left, c_right = st.columns(2)
                    with c_left:
                        if st.button("🚨 Confirmar Evolução: Grave", key=f"btn_grave_{id_hist}", use_container_width=True):
                            st.query_params["fb_id"] = str(id_hist)
                            st.query_params["fb_grave"] = "1"
                            st.rerun()
                    with c_right:
                        if st.button("✅ Confirmar Evolução: Leve/Mod.", key=f"btn_leve_{id_hist}", use_container_width=True):
                            st.query_params["fb_id"] = str(id_hist)
                            st.query_params["fb_grave"] = "0"
                            st.rerun()
                            
            # Pequeno espaçamento externo entre um card e o próximo da listagem
            st.markdown('<div style="margin-bottom: 4px;"></div>', unsafe_allow_html=True)
            
    # Botão de escape global para retornar ao início do app
    st.write("")
    if st.button("← Voltar", use_container_width=True, key="btn_voltar_historico"):
        st.session_state.step = "form"
        st.rerun()

    render_footer()