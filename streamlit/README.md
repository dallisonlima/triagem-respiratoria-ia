# 🩺 MedTriagem IA — Dashboard Streamlit

Interface visual interativa do sistema de **triagem clínica respiratória**, construída com **Streamlit**. Este painel consome a API FastAPI do backend para treinar e executar o modelo de Inteligência Artificial, além de interagir diretamente com o **Supabase** para autenticação de usuários, armazenamento de histórico e registro de feedbacks médicos.

---

## 📋 Visão Geral

O dashboard é dividido em **cinco telas**:

| Tela | Descrição |
|------|-----------| 
| 🔐 **Login / Cadastro** | Autenticação via Supabase Auth (email + senha), com sessão persistida via cookie |
| 🚀 **Treinamento** | Dispara o treinamento da IA com dados reais do SIVEP-Gripe + feedbacks via backend |
| 📝 **Formulário** | Coleta dados do paciente (Nome, CPF, Sexo, Nascimento) e sintomas/comorbidades |
| 📊 **Resultado** | Exibe o diagnóstico com nível de risco, barra de probabilidade, limiares e perfil |
| 📋 **Histórico** | Lista as últimas 50 avaliações do usuário com opção de feedback médico |

---

## 🗂️ Estrutura de Arquivos

```
streamlit/
├── app.py            # Aplicação principal do Streamlit (auth, formulário, histórico, feedback)
├── style.css         # Design system externo (tema, componentes, responsividade)
├── requirements.txt  # Dependências Python do módulo
├── venv/             # Ambiente virtual (gerado localmente, não versionado)
└── README.md         # Este arquivo
```

---

## ⚙️ Pré-requisitos

- Python **3.10+**
- Backend FastAPI rodando em `http://localhost:8000` *(obrigatório)*

> ⚠️ **Importante:** O Streamlit depende inteiramente do backend para treinar o modelo e gerar previsões. Certifique-se de que a API está ativa antes de iniciar o dashboard.

---

## 🚀 Como Rodar

### 1. Entre na pasta do módulo

```bash
cd streamlit
```

### 2. Crie e ative o ambiente virtual

> Faça isso apenas na **primeira vez** ou se a pasta `venv/` não existir.

```bash
# Criar o venv
python3 -m venv venv

# Ativar (Linux / macOS)
source venv/bin/activate

# Ativar (Windows)
venv\Scripts\activate
```

### 3. Configure as variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto (ao lado do `backend/`) com as seguintes variáveis:

```env
SUPABASE_URL="https://xxxx.supabase.co"
SUPABASE_KEY="sua_chave_aqui"
TRAIN_SECRET="sua_chave_secreta_aqui"

# Opcionais (possuem valores padrão para desenvolvimento local)
ALLOWED_ORIGINS="*"
API_BASE="http://localhost:8000"
```

> O `TRAIN_SECRET` deve ser idêntico ao configurado no backend. Sem ele, o endpoint `/train` retornará erro 403.

### 4. Instale as dependências

Com o venv ativo, instale os pacotes necessários:

```bash
pip install -r requirements.txt
```

### 5. Inicie o dashboard

```bash
streamlit run app.py
```

O painel estará disponível em: **http://localhost:8501**

---

## 🔄 Nas próximas vezes

Nas execuções seguintes, basta ativar o venv e rodar:

```bash
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
streamlit run app.py
```

---

## 📦 Dependências (`requirements.txt`)

| Pacote | Função |
|--------|--------|
| `streamlit` | Framework para criação do dashboard interativo |
| `requests` | Comunicação HTTP com a API FastAPI e Supabase REST API |
| `python-dotenv` | Leitura das variáveis de ambiente do arquivo `.env` |

---

## 🎨 Design System (`style.css`)

O design visual está completamente **externalizado** em `style.css`, separando responsabilidades: o `app.py` contém apenas lógica Python, enquanto o CSS concentra todo o sistema de design.

O arquivo é carregado dinamicamente no início da aplicação:

```python
def _load_css(path: str):
    with open(path, encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

_load_css(os.path.join(os.path.dirname(__file__), "style.css"))
```

**Características do tema:**
- Fundo light premium com gradiente suave (`#f8fbff → #eef5ff`)
- Tipografia: `Inter` (interface) + `JetBrains Mono` (dados técnicos)
- Componentes: header flutuante, badge animado, cards com acento colorido, tags pill
- Responsivo para telas a partir de 768px

---

## 🔐 Autenticação (Supabase Auth)

O dashboard implementa um sistema completo de **login e cadastro** via Supabase Auth REST API:

- **Login**: `POST /auth/v1/token?grant_type=password` — retorna JWT (`access_token`)
- **Cadastro**: `POST /auth/v1/signup` — cria conta e redireciona para login
- **Sessão**: O token e email do usuário são armazenados em `st.session_state` e persistidos via **cookie no navegador** (validade de 7 dias), evitando re-login a cada reload
- **Logout**: Limpa a sessão e remove o cookie

```python
def _supabase_login(email: str, password: str) -> dict:
    resp = requests.post(
        f"{SUPABASE_URL}/auth/v1/token?grant_type=password",
        headers=AUTH_HEADERS,
        json={"email": email, "password": password},
        timeout=10,
    )
    return resp.json(), resp.status_code
```

Mensagens de erro do Supabase são traduzidas automaticamente para português.

---

## ⚙️ Configuração de Ambiente

As variáveis sensíveis são lidas com prioridade via `os.environ` (`.env`) e fallback para `st.secrets` (produção no Streamlit Cloud):

```python
def _secret(key: str, default: str) -> str:
    if val := os.environ.get(key):
        return val
    try:
        return st.secrets[key]
    except Exception:
        return default

API_BASE     = _secret("API_BASE",     "http://localhost:8000")
TRAIN_SECRET = _secret("TRAIN_SECRET", "")
```

Em produção, defina as variáveis em `~/.streamlit/secrets.toml` ou nas configurações do Streamlit Cloud.

---

## 🧠 Fluxo da Aplicação

```
Usuário abre o dashboard
        │
        ▼
[Tela 0] Login / Cadastro
  • Email + Senha via Supabase Auth
  • Sessão persistida via cookie (7 dias)
        │
        ▼
[Tela 1] Clica em "Iniciar Treinamento da IA"
        │
        ▼
Streamlit faz POST /train
  Header: X-Train-Secret → FastAPI autentica
        │  (dados SIVEP-Gripe + feedbacks via Supabase)
        ▼
Modelo LogisticRegression treinado
Resultado em cache por 3 horas ⚡
        │
        ▼
[Tela 2] Formulário do Paciente
  • Dados obrigatórios: Nome, CPF (com máscara e validação JS),
    Sexo, Data de Nascimento (calcula idade automaticamente)
  • Sintomas: Febre, Tosse, Falta de Ar, Dor de Garganta, Sat. O₂ < 95%
  • Comorbidades: Asma, Diabetes, Cardiopatia
  ⚠️ Todos os dados básicos + ao menos 1 sintoma/comorbidade obrigatório
        │
        ▼
Streamlit faz POST /predict → FastAPI
  (todos os perfis enviados ao modelo, sem pré-filtro no frontend)
        │
        ▼
[Tela 3] Resultado do Diagnóstico
  🚨 Grave    → prob ≥ 60%
  ⚠️ Moderado → prob ≥ 30%
  ✅ Leve     → prob < 30%
  (limiares definidos e retornados pelo backend)
        │  (salva automaticamente no Supabase → tabela `historico`)
        ▼
[Tela 4] Histórico de Avaliações
  • Últimas 50 avaliações do usuário logado
  • Cards com badge de classificação, dados do paciente e sintomas
  • Botões de feedback médico: confirmar desfecho real (Grave / Leve)
  • Feedbacks validados são marcados visualmente
        │
        ▼
[Retroalimentação → próximo ciclo de treinamento incorpora os feedbacks ↺]
```

---

## 📋 Formulário do Paciente

O formulário coleta os seguintes dados, todos com validação:

### Dados Básicos (obrigatórios)

| Campo | Tipo | Validação |
|-------|------|-----------|
| **Nome** | Texto livre | Não pode ser vazio |
| **CPF** | Texto com máscara | Máscara `000.000.000-00` aplicada via JS em tempo real + validação de dígitos verificadores (algoritmo completo no Python e no JS) |
| **Sexo** | Selectbox | Feminino / Masculino — seleção obrigatória |
| **Nascimento** | Date picker | Formato DD/MM/AAAA — calcula idade automaticamente |

### Sintomas e Comorbidades (ao menos 1 obrigatório)

| Sintomas | Comorbidades |
|----------|-------------|
| 🌡️ Febre | 🌬️ Asma |
| 😮‍💨 Tosse | 💉 Diabetes |
| 💨 Falta de Ar / Dispneia | ❤️ Cardiopatia |
| 🔴 Dor de Garganta | |
| 🩸 Saturação O₂ < 95% | |

O botão "Analisar com IA" permanece **desabilitado** até que todos os dados básicos estejam preenchidos e ao menos um sintoma/comorbidade esteja marcado.

---

## 📋 Histórico e Feedback Médico

### Histórico

Cada avaliação realizada é automaticamente salva na tabela `historico` do Supabase, vinculada ao email do usuário logado. O histórico exibe:

- Badge de classificação com cor (Grave / Moderado / Leve)
- Nome do paciente (quando disponível)
- Data e hora da avaliação
- Idade, sexo e CPF
- Tags dos sintomas/comorbidades selecionados
- Percentual de probabilidade de complicação

### Feedback Médico (Retroalimentação)

Cada card do histórico que **ainda não foi validado** exibe dois botões:

- **"Confirmar Evolução: Grave (UTI/Óbito)"** — registra que o paciente realmente evoluiu para quadro grave
- **"Confirmar Evolução: Leve/Moderado"** — registra que o paciente não apresentou complicação

O feedback é salvo na tabela `feedbacks` do Supabase e, no próximo ciclo de treinamento, é incorporado ao dataset para melhorar a acurácia do modelo.

Cards já validados exibem um badge verde: `✅ Validado pelo médico como: [desfecho]`.

---

## 🔗 Endpoints e APIs Consumidos

### API FastAPI (Backend)

| Método | Endpoint | Header obrigatório | Descrição |
|--------|----------|--------------------|-----------| 
| `POST` | `/train` | `X-Train-Secret` | Treina o modelo com os dados do Supabase |
| `POST` | `/predict` | — | Retorna probabilidade, classificação e limiares |

### Supabase REST API (acesso direto)

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| `POST` | `/auth/v1/token?grant_type=password` | Login (retorna JWT) |
| `POST` | `/auth/v1/signup` | Cadastro de novo usuário |
| `POST` | `/rest/v1/historico` | Salva avaliação no histórico |
| `GET` | `/rest/v1/historico?...` | Busca histórico do usuário logado (com join feedbacks) |
| `POST` | `/rest/v1/feedbacks` | Registra feedback médico |

**Resposta do `/predict`:**
```json
{
  "probabilidadeGravidade": 32.7,
  "classificacao": "moderado",
  "limiares": {
    "grave": 60.0,
    "moderado": 30.0
  }
}
```

> A classificação e os limiares são definidos **exclusivamente pelo backend**. O frontend apenas exibe o que recebe, sem recalcular nem filtrar.

---

## ⚡ Cache de Performance

O Streamlit utiliza `@st.cache_data` com TTL de **3 horas** para o resultado do treinamento, evitando chamadas desnecessárias ao backend. O `secret` é incluído como parâmetro da função cacheada para garantir invalidação automática em caso de mudança de credencial:

```python
@st.cache_data(ttl=10800, show_spinner=False)
def fetch_and_train(secret: str):
    headers = {"X-Train-Secret": secret} if secret else {}
    r = requests.post(f"{API_BASE}/train", headers=headers, timeout=300)
    r.raise_for_status()
    return r.json()
```

Para forçar um novo treinamento antes do cache expirar, reinicie o servidor Streamlit.

---

## 🐛 Problemas Comuns

| Erro | Causa Provável | Solução |
|------|----------------|---------| 
| `command not found: streamlit` | Venv não ativado ou deps não instaladas | Ative o venv e rode `pip install -r requirements.txt` |
| `Erro de conexão: ...` | Backend FastAPI não está rodando | Suba a API antes de iniciar o Streamlit |
| `403 Forbidden` no `/train` | `TRAIN_SECRET` ausente ou divergente do backend | Verifique se o `.env` tem o mesmo valor em ambos os módulos |
| `Erro do Backend: ...` | Problema interno na API (ex: Supabase offline) | Verifique as variáveis de ambiente `.env` e os logs do backend |
| `StreamlitSecretNotFoundError` | `st.secrets` acessado sem `secrets.toml` | Normal em dev local — o código faz fallback para `os.environ` automaticamente |
| `externally-managed-environment` | Python gerenciado pelo sistema | Use sempre o ambiente virtual (`venv`) |

---

## 🏗️ Arquitetura Completa

Este módulo faz parte de um projeto full-stack. Para rodar o sistema completo:

| Serviço | Comando | Porta |
|---------|---------|-------|
| **Backend (FastAPI)** | `uvicorn backend.api:app --reload` | `:8000` |
| **Streamlit** | `streamlit run app.py` | `:8501` |

Consulte o [README principal](../README.md) para instruções completas de configuração do ambiente.

---

*⚕️ Ferramenta de apoio clínico — não substitui avaliação médica profissional.*  
*Dados: SIVEP-Gripe (DataSUS) · Modelo: Scikit-Learn LogisticRegression · Backend: FastAPI*