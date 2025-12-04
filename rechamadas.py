import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io # Para lidar com arquivos em memória

warnings.filterwarnings('ignore') # Ignora avisos do pandas/matplotlib que não impedem a execução

# --- CONFIGURAÇÕES GLOBAIS ---
VALOR_LIGACAO = 7.56
MIN_LIGACOES_GRAF = 2   # Mínimo de ligações para um telefone aparecer no gráfico de reincidência

# --- 1. FUNÇÕES AUXILIARES ---
def converter_duracao_para_segundos(duracao_str):
    """Converte string de duração (mm:ss ou hh:mm:ss) para segundos."""
    try:
        duracao_str = str(duracao_str).strip()
        if ':' in duracao_str:
            partes = list(map(int, duracao_str.split(':')))
            if len(partes) == 2:  # mm:ss
                return partes[0] * 60 + partes[1]
            elif len(partes) == 3:  # hh:mm:ss
                return partes[0] * 3600 + partes[1] * 60 + partes[2]
        else: # Já é um número (segundos)
            return int(float(duracao_str))
    except ValueError:
        return 0 # Retorna 0 se não conseguir converter

# Simplificando detect_datetime_column para focar apenas em nomes de colunas
def detect_datetime_column(df):
    """
    Detecta a coluna de data/hora no DataFrame com base em nomes comuns.
    Retorna o nome da coluna detectada ou None.
    """
    # Priorizar a coluna 'Data' se ela existir e não for a mesma que 'data_hora'
    if 'Data' in df.columns and 'data_hora' not in df.columns:
        return 'Data'
    for col in df.columns:
        col_lower = col.lower()
        # Prioriza nomes mais específicos e depois os genéricos
        if 'data_hora' == col_lower: return col
        if 'datetime' == col_lower: return col
        if 'timestamp' == col_lower: return col
        if 'Data' == col_lower: return col
        if 'hora' == col_lower: return col
        if 'time' == col_lower: return col
    return None

# --- 2. CARREGAMENTO E PREPARAÇÃO DOS DADOS (Adaptado para Streamlit e múltiplas abas) ---
@st.cache_data(show_spinner="Carregando e processando dados...")
def analisar_ligacoes_callcenter_streamlit(uploaded_file):
    """
    Carrega e prepara os dados de ligações de call center de um arquivo uploaded (CSV ou Excel).
    Detecta automaticamente colunas de data/hora, telefone e duração.
    Processa todas as abas se for Excel.
    """
    if uploaded_file is None:
        return None, "Por favor, faça o upload de um arquivo para começar a análise."

    file_details = {"filename": uploaded_file.name, "filetype": uploaded_file.type, "filesize": uploaded_file.size}
    st.write(f"📖 Carregando arquivo: **{file_details['filename']}** ({file_details['filesize'] / 1024:.2f} KB)")

    dfs = []
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension == 'csv':
        # Tentar diferentes codificações e separadores para CSV
        df_temp = None
        # Para CSVs, precisamos ler o conteúdo em memória para tentar diferentes encodings/seps
        uploaded_file_content = uploaded_file.getvalue()

        for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
            for sep in [',', ';', '\t']:
                try:
                    # Decodifica o conteúdo e usa io.StringIO para simular um arquivo
                    df_temp = pd.read_csv(io.StringIO(uploaded_file_content.decode(encoding)), sep=sep)
                    if not df_temp.empty and len(df_temp.columns) > 1:
                        st.info(f"   - Sucesso com encoding: {encoding}, separador: '{sep}'")
                        break # Sai do loop de separadores
                except Exception:
                    continue # Tenta o próximo separador/encoding
            if df_temp is not None and not df_temp.empty and len(df_temp.columns) > 1:
                break # Sai do loop de encodings

        if df_temp is None or df_temp.empty or len(df_temp.columns) <= 1:
            st.error(f"   ❌ Não foi possível carregar o arquivo CSV corretamente. Verifique o formato.")
            return None, "Erro ao carregar CSV."
        dfs.append(df_temp)

    elif file_extension in ['xlsx', 'xls']:
        try:
            uploaded_file.seek(0) # Garante que o ponteiro está no início do arquivo
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            st.info(f"   - Arquivo Excel detectado com {len(sheet_names)} abas: {', '.join(sheet_names)}")

            for sheet_name in sheet_names:
                st.write(f"     Processando aba: **{sheet_name}**")
                df_temp = pd.read_excel(excel_file, sheet_name=sheet_name)
                if not df_temp.empty and len(df_temp.columns) > 1:
                    dfs.append(df_temp)
                else:
                    st.warning(f"     Aba '{sheet_name}' está vazia ou tem poucas colunas. Ignorando.")
        except Exception as e:
            st.error(f"   ❌ Erro ao carregar arquivo Excel: {e}")
            return None, "Erro ao carregar Excel."
    else:
        st.error("   ❌ Formato de arquivo não suportado. Por favor, faça upload de um arquivo CSV ou Excel (.xlsx, .xls).")
        return None, "Formato de arquivo inválido."

    if not dfs:
        return None, "Nenhum dado válido foi carregado de nenhuma aba/arquivo."

    # Concatenar todos os dataframes
    df_completo = pd.concat(dfs, ignore_index=True)
    st.success(f"✅ Total de registros combinados de todas as abas: **{len(df_completo):,}**")

    # Remover colunas 'Unnamed: X'
    df_completo = df_completo.loc[:, ~df_completo.columns.str.contains('^Unnamed', na=False)]

    # Detectar e padronizar nomes de colunas
    col_map = {}
    found_data_hora, found_telefone, found_duracao = False, False, False

    # Detectar coluna de data/hora
    datetime_col_name = detect_datetime_column(df_completo)
    if datetime_col_name:
        col_map[datetime_col_name] = 'data_hora'
        found_data_hora = True
        st.info(f"   - Coluna de data/hora detectada: '{datetime_col_name}' (renomeada para 'data_hora')")
    else:
        st.warning("   - Nenhuma coluna de data/hora com nome comum foi detectada automaticamente.")

    for col in df_completo.columns:
        col_lower = col.lower()

        # TELEFONE: incluir 'ani' como sinônimo
        if not found_telefone and (
            any(k in col_lower for k in ['telefone', 'phone', 'numero', 'fone', 'tel'])
            or col_lower == 'ani'
        ):
            col_map[col] = 'telefone'
            found_telefone = True
            st.info(f"   - Coluna de telefone detectada: '{col}' (renomeada para 'telefone')")

        # DURAÇÃO: incluir 'duração' com acento
        elif not found_duracao and (
            any(k in col_lower for k in ['duracao', 'duração', 'duration', 'tempo'])
        ):
            col_map[col] = 'duracao'
            found_duracao = True
            st.info(f"   - Coluna de duração detectada: '{col}' (renomeada para 'duracao')")



    df_completo = df_completo.rename(columns=col_map)

    if not found_data_hora or not found_telefone:
        st.error(f"   ❌ Não foi possível identificar as colunas 'data_hora' ou 'telefone'. Colunas disponíveis: {list(df_completo.columns)}. Verifique seu arquivo.")
        return None, "Colunas essenciais não encontradas."

    # --- Processamento de dados ---
    # Garantir que 'data_hora' é string e remover espaços em branco
    df_completo['data_hora'] = df_completo['data_hora'].astype(str).str.strip()

    # Limpeza adicional: remover caracteres que podem atrapalhar a conversão
    # Ex: espaços duplos, quebras de linha, etc.
    df_completo['data_hora'] = df_completo['data_hora'].str.replace(r'\s+', ' ', regex=True)

    # Converter 'data_hora' para datetime
    st.write("🔄 Convertendo coluna 'data_hora' para datetime...")
    # Formatos de data/hora, priorizando o formato 'dd/MM/yyyy HH:mm:ss'
    datetime_formats = [
        '%d/%m/%Y %H:%M:%S',      # SEU FORMATO ESPECÍFICO - PRIORIDADE MÁXIMA
        '%d/%m/%Y %H:%M',         # Seu formato sem segundos
        '%Y-%m-%dT%H:%M:%S',      # ISO com T
        '%Y-%m-%d %H:%M:%S',      # ISO com espaço
        '%Y-%m-%d %H:%M',         # ISO sem segundos
        '%Y-%m-%d',               # ISO só data
        '%d/%m/%Y',               # BR só data
        '%m/%d/%Y',               # US só data
    ]

    df_completo['datetime'] = pd.NaT # Inicializa com Not a Time

    # Tentar converter em blocos para otimizar e dar feedback
    total_rows = len(df_completo)
    converted_count = 0

    for fmt in datetime_formats:
        mask = df_completo['datetime'].isna()
        if not mask.any(): # Se não há mais NaT, todas foram convertidas
            break

        # Tenta converter apenas as linhas que ainda não foram convertidas
        df_completo.loc[mask, 'datetime'] = pd.to_datetime(df_completo.loc[mask, 'data_hora'], format=fmt, errors='coerce')

        newly_converted = (~df_completo.loc[mask, 'datetime'].isna()).sum()
        if newly_converted > 0:
            converted_count += newly_converted
            st.info(f"   - Converteu {newly_converted:,} datas com o formato '{fmt}'. Total convertido: {converted_count:,}/{total_rows:,}")

    # Última tentativa: inferência automática se ainda houver nulos
    mask_final = df_completo['datetime'].isna()
    if mask_final.any():
        st.warning(f"   - Ainda há {mask_final.sum():,} datas inválidas. Tentando inferência automática de formato (dayfirst=True)...")
        df_completo.loc[mask_final, 'datetime'] = pd.to_datetime(df_completo.loc[mask_final, 'data_hora'], dayfirst=True, errors='coerce')
        final_converted = (~df_completo.loc[mask_final, 'datetime'].isna()).sum()
        if final_converted > 0:
            converted_count += final_converted
            st.info(f"   - Converteu {final_converted:,} datas por inferência automática. Total convertido: {converted_count:,}/{total_rows:,}")


    registros_antes = len(df_completo)
    df_completo = df_completo.dropna(subset=['datetime'])
    if len(df_completo) < registros_antes:
        st.warning(f"⚠️ Removidos {registros_antes - len(df_completo)} registros com data/hora inválida após todas as tentativas de conversão.")

    if df_completo.empty:
        st.error("🚨 Todos os registros foram removidos devido a datas/horas inválidas. Verifique o formato da coluna 'data_hora' nos seus arquivos.")
        return None, "Dados vazios após limpeza de datas."

    # Limpar e validar 'telefone'
    df_completo['telefone'] = df_completo['telefone'].astype(str).str.replace(r'[^\d]', '', regex=True)
    registros_antes = len(df_completo)
    df_completo = df_completo[df_completo['telefone'].str.len() >= 8]
    if len(df_completo) < registros_antes:
        st.warning(f"⚠️ Removidos {registros_antes - len(df_completo)} registros com telefone inválido (menos de 8 dígitos).")

    # Converter 'duracao' para segundos
    if 'duracao' in df_completo.columns:
        df_completo['duracao_segundos'] = df_completo['duracao'].apply(converter_duracao_para_segundos)
    else:
        df_completo['duracao_segundos'] = 0 # Se não houver coluna de duração, assume 0
        st.warning("   - Coluna 'duracao' não encontrada. Duração das ligações será considerada 0.")

    # Ordenar para análise de rechamadas
    df_completo = df_completo.sort_values(['telefone', 'datetime']).reset_index(drop=True)
    st.success(f"\n✅ Dados processados com sucesso: **{len(df_completo):,}** ligações. Período: **{df_completo['datetime'].min():%d/%m/%Y}** a **{df_completo['datetime'].max():%d/%m/%Y}**")

    # Exibir o número de atendimentos na carga de dados
    st.metric("Total de Atendimentos (Ligações) Carregados", f"{len(df_completo):,}")

    return df_completo, None

# --- 3. FUNÇÕES DE ANÁLISE (Mantidas como estão, pois operam no DataFrame processado) ---
def identificar_faixas_rechamada(df):
    """Identifica rechamadas em faixas de 0-24h, 24-48h, 48-72h."""
    rechamadas = {'0-24h': [], '24-48h': [], '48-72h': []}
    for telefone, grupo in df.groupby('telefone'):
        if len(grupo) < 2: continue
        grupo = grupo.sort_values('datetime')
        datas = grupo['datetime'].values
        duras = grupo['duracao_segundos'].values
        for i in range(1, len(datas)):
            diff_h = (datas[i] - datas[i-1]) / np.timedelta64(1, 'h')
            rec = {
                'telefone': telefone,
                'primeira_ligacao': datas[i-1],
                'segunda_ligacao': datas[i],
                'diferenca_horas': float(diff_h),
                'duracao_primeira_seg': duras[i-1],
                'duracao_segunda_seg': duras[i]
            }
            if diff_h <= 24: rechamadas['0-24h'].append(rec)
            elif 24 < diff_h <= 48: rechamadas['24-48h'].append(rec)
            elif 48 < diff_h <= 72: rechamadas['48-72h'].append(rec)
    return rechamadas

def faixas_ligacoes_e_reincidentes(df):
    """Calcula a contagem de ligações por telefone e as faixas de reincidência."""
    contagem_por_telefone = df.groupby("telefone").size()
    faixas = {
        '1 ligação': len(contagem_por_telefone[contagem_por_telefone == 1]),
        '2-5 ligações': len(contagem_por_telefone[(contagem_por_telefone >= 2) & (contagem_por_telefone <= 5)]),
        '6-10 ligações': len(contagem_por_telefone[(contagem_por_telefone >= 6) & (contagem_por_telefone <= 10)]),
        '11-20 ligações': len(contagem_por_telefone[(contagem_por_telefone >= 11) & (contagem_por_telefone <= 20)]),
        '21-50 ligações': len(contagem_por_telefone[(contagem_por_telefone >= 21) & (contagem_por_telefone <= 50)]),
        'Mais de 50 ligações': len(contagem_por_telefone[contagem_por_telefone > 50])
    }
    telefones_ligaram_mais_de_uma_vez = len(contagem_por_telefone[contagem_por_telefone > 1])
    return faixas, telefones_ligaram_mais_de_uma_vez, contagem_por_telefone

def clientes_frequentes(df, N=50):
    """Identifica os N clientes que mais ligaram, com detalhes."""
    contagem = df.groupby('telefone').agg(
        total_ligacoes=('datetime', 'count'),
        primeira_ligacao=('datetime', 'min'),
        ultima_ligacao=('datetime', 'max'),
        duracao_total_seg=('duracao_segundos', 'sum'),
        duracao_media_seg=('duracao_segundos', 'mean')
    ).round(2)
    contagem['periodo_atividade_dias'] = (contagem['ultima_ligacao'] - contagem['primeira_ligacao']).dt.days
    contagem['frequencia_ligacoes_por_dia'] = contagem.apply(
        lambda x: x['total_ligacoes'] / max(1, x['periodo_atividade_dias']) if x['periodo_atividade_dias'] > 0 else x['total_ligacoes'], axis=1
    ).round(2)
    contagem['duracao_total_min'] = (contagem['duracao_total_seg'] / 60).round(2)
    contagem['duracao_media_min'] = (contagem['duracao_media_seg'] / 60).round(2)
    return contagem.sort_values('total_ligacoes', ascending=False).head(N)

def calcular_impacto_financeiro(rechamadas, valor_ligacao=VALOR_LIGACAO):
    """Calcula o impacto financeiro das rechamadas."""
    impacto_por_faixa = {k: len(v) * valor_ligacao for k, v in rechamadas.items()}
    total_religacoes = sum(len(v) for v in rechamadas.values())
    return {
        'total_religacoes': total_religacoes,
        'impacto_total': total_religacoes * valor_ligacao,
        'valor_por_ligacao': valor_ligacao,
        'impacto_por_faixa': impacto_por_faixa
    }

def gerar_consolidado(df, rechamadas, clientes_frequentes_df, faixas_ligacoes, telefones_reincidentes, impacto_financeiro):
    """Gera um dicionário consolidado com todas as métricas para relatórios."""
    dias_pt = {'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta', 
               'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
    stats = {
        'total_ligacoes': len(df),
        'clientes_unicos': df['telefone'].nunique(),
        'media_ligacoes_por_cliente': round(len(df) / df['telefone'].nunique(), 2),
        'periodo_analise': f"{df['datetime'].min():%d/%m/%Y} a {df['datetime'].max():%d/%m/%Y}",
        'duracao_total_horas': round(df['duracao_segundos'].sum() / 3600, 2),
        'duracao_media_minutos': round(df['duracao_segundos'].mean() / 60, 2)
    }
    df['dia_semana'] = df['datetime'].dt.day_name().map(dias_pt)
    ligacoes_por_dia = df['dia_semana'].value_counts().to_dict()
    df['hora'] = df['datetime'].dt.hour
    horarios_pico = df['hora'].value_counts().head(5).to_dict()
    religacoes_resumo = {
        faixa: {
            'quantidade': len(dados),
            'clientes_unicos': len(set(item['telefone'] for item in dados)) if dados else 0,
            'tempo_medio_horas': float(np.mean([item['diferenca_horas'] for item in dados]) if dados else 0),
            'impacto_financeiro': impacto_financeiro['impacto_por_faixa'][faixa]
        } for faixa, dados in rechamadas.items()
    }
    top_10_clientes = clientes_frequentes_df.head(10)[['total_ligacoes', 'frequencia_ligacoes_por_dia', 'duracao_total_min']].to_dict(orient='index')
    return {
        'estatisticas_gerais': stats,
        'ligacoes_por_dia': ligacoes_por_dia,
        'horarios_pico': {f'{int(k)}h': int(v) for k, v in horarios_pico.items()},
        'religacoes': religacoes_resumo,
        'top_clientes': top_10_clientes,
        'faixas_ligacoes': faixas_ligacoes,
        'telefones_reincidentes': telefones_reincidentes,
        'impacto_financeiro': impacto_financeiro
    }

# --- 4. FUNÇÕES DE VISUALIZAÇÃO E RELATÓRIOS (Adaptadas para Streamlit) ---
def plot_reincidentes_streamlit(contagem_por_telefone, min_ligacoes=MIN_LIGACOES_GRAF):
    """
    Gera um histograma da quantidade de ligações por telefone para Streamlit.
    """
    reincidentes_filtrados = contagem_por_telefone[contagem_por_telefone >= min_ligacoes]
    if reincidentes_filtrados.empty:
        st.warning(f"Nenhum telefone com {min_ligacoes} ou mais ligações para exibir no gráfico de reincidência.")
        return None

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(reincidentes_filtrados, bins=range(min_ligacoes, reincidentes_filtrados.max() + 2), kde=False, color='navy', ax=ax)
    ax.set_title(f'Distribuição de Telefones por Número de Ligações (≥{min_ligacoes})')
    ax.set_xlabel('Número de Ligações')
    ax.set_ylabel('Quantidade de Telefones')
    ax.set_xticks(range(min_ligacoes, reincidentes_filtrados.max() + 2))
    plt.tight_layout()
    return fig

def create_dashboard_plots(consolidado, reincidentes_serie, min_ligacoes_graf):
    """Cria os gráficos individuais para o dashboard Streamlit."""
    plt.style.use('seaborn-v0_8-whitegrid')

    plots = {}

    # 1. Ligações por Dia da Semana
    fig_dia, ax_dia = plt.subplots(figsize=(8, 4))
    dias = list(consolidado['ligacoes_por_dia'].keys())
    valores_dia = list(consolidado['ligacoes_por_dia'].values())
    bars_dia = ax_dia.bar(dias, valores_dia, color='skyblue', edgecolor='navy')
    ax_dia.set_title('📅 Ligações por Dia da Semana', fontsize=12)
    ax_dia.tick_params(axis='x', rotation=45, labelsize=10)
    ax_dia.tick_params(axis='y', labelsize=10)
    for bar in bars_dia: ax_dia.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plots['ligacoes_por_dia'] = fig_dia

    # 2. Horários de Pico
    fig_hora, ax_hora = plt.subplots(figsize=(8, 4))
    horas = list(consolidado['horarios_pico'].keys())
    valores_hora = list(consolidado['horarios_pico'].values())
    bars_hora = ax_hora.bar(horas, valores_hora, color='lightgreen', edgecolor='darkgreen')
    ax_hora.set_title('🕐 Top 5 Horários de Pico', fontsize=12)
    ax_hora.tick_params(axis='x', labelsize=10)
    ax_hora.tick_params(axis='y', labelsize=10)
    for bar in bars_hora: ax_hora.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plots['horarios_pico'] = fig_hora

    # 3. Faixas de Ligações
    fig_faixas, ax_faixas = plt.subplots(figsize=(8, 8))
    faixas_labels = [k for k, v in consolidado['faixas_ligacoes'].items() if v > 0]
    faixas_sizes = [v for v in consolidado['faixas_ligacoes'].values() if v > 0]
    if faixas_sizes:
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(faixas_sizes)))
        ax_faixas.pie(faixas_sizes, labels=faixas_labels, autopct='%1.1f%%', colors=colors, startangle=90, textprops={'fontsize': 10})
        ax_faixas.set_title('📊 Distribuição por Faixas de Ligações', fontsize=12)
    else:
        ax_faixas.text(0.5, 0.5, 'Nenhuma faixa de ligação para exibir', horizontalalignment='center', verticalalignment='center', transform=ax_faixas.transAxes, fontsize=10)
        ax_faixas.axis('off')
    plt.tight_layout()
    plots['faixas_ligacoes'] = fig_faixas

    # 4. Rechamadas por Período
    fig_relig, ax_relig = plt.subplots(figsize=(8, 4))
    periodos = ['0-24h', '24-48h', '48-72h']
    qtd_religacoes = [consolidado['religacoes'][p]['quantidade'] for p in periodos]
    bars_relig = ax_relig.bar(periodos, qtd_religacoes, color=['red', 'orange', 'gold'], edgecolor='black')
    ax_relig.set_title('📞 Rechamadas por Período', fontsize=12)
    ax_relig.tick_params(axis='x', labelsize=10)
    ax_relig.tick_params(axis='y', labelsize=10)
    for bar in bars_relig: ax_relig.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plots['rechamadas_por_periodo'] = fig_relig

    # 5. Histograma de Reincidência
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    if not reincidentes_serie.empty:
        sns.histplot(reincidentes_serie, bins=range(min_ligacoes_graf, reincidentes_serie.max() + 2), kde=False, ax=ax_hist, color='darkorchid')
        ax_hist.set_title(f'📉 Reincidência (Telefones com ≥{min_ligacoes_graf} Ligações)', fontsize=12)
        ax_hist.set_xlabel('Número de Ligações', fontsize=10)
        ax_hist.set_ylabel('Quantidade de Telefones', fontsize=10)
        ax_hist.tick_params(axis='x', labelsize=9)
        ax_hist.tick_params(axis='y', labelsize=9)
    else:
        ax_hist.text(0.5, 0.5, 'Nenhum telefone com ≥2 ligações', horizontalalignment='center', verticalalignment='center', transform=ax_hist.transAxes, fontsize=10)
        ax_hist.axis('off')
    plt.tight_layout()
    plots['hist_reincidencia'] = fig_hist

    # 6. Top 10 Clientes que Mais Ligaram
    fig_top_clientes, ax_top_clientes = plt.subplots(figsize=(8, 4))
    top_clientes_df = pd.DataFrame(consolidado['top_clientes']).T
    if not top_clientes_df.empty:
        bars_top = ax_top_clientes.bar(top_clientes_df.index.astype(str), top_clientes_df['total_ligacoes'], color='teal', edgecolor='black')
        ax_top_clientes.set_title('🏆 Top 10 Clientes que Mais Ligaram', fontsize=12)
        ax_top_clientes.set_xlabel('Telefone', fontsize=10)
        ax_top_clientes.set_ylabel('Total de Ligações', fontsize=10)
        ax_top_clientes.tick_params(axis='x', labelsize=9, rotation=45) 
        ax_top_clientes.set_xticklabels(top_clientes_df.index.astype(str), rotation=45, ha='right', fontsize=9)
        ax_top_clientes.tick_params(axis='y', labelsize=9)
        for bar in bars_top: ax_top_clientes.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=7)
    else:
        ax_top_clientes.text(0.5, 0.5, 'Nenhum cliente frequente encontrado', horizontalalignment='center', verticalalignment='center', transform=ax_top_clientes.transAxes, fontsize=10)
        ax_top_clientes.axis('off')
    plt.tight_layout()
    plots['top_clientes'] = fig_top_clientes

    return plots

def to_excel_buffer(df, rechamadas_detalhe, clientes_frequentes_todos, consolidado, faixas_ligacoes, contagem_por_telefone_bruta, reincidentes_serie_filtrada):
    """
    Salva todos os resultados em um único arquivo Excel com múltiplas abas em um buffer de memória.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # --- ABA: Consolidado Geral ---
        stats = consolidado['estatisticas_gerais']
        impacto = consolidado['impacto_financeiro']
        consolidado_data = [
            ['Total de Ligações', stats['total_ligacoes']],
            ['Clientes Únicos', stats['clientes_unicos']],
            ['Números que ligaram mais de uma vez', consolidado['telefones_reincidentes']],
            ['Média Ligações/Cliente', stats['media_ligacoes_por_cliente']],
            ['Período de Análise', stats['periodo_analise']],
            ['Duração Total (horas)', stats['duracao_total_horas']],
            ['Duração Média por Ligação (min)', stats['duracao_media_minutos']],
            ['', ''], # Linha em branco
            ['IMPACTO FINANCEIRO', ''],
            ['Total de Rechamadas', impacto['total_religacoes']],
            ['Custo por Ligação', f"R$ {impacto['valor_por_ligacao']:.2f}"],
            ['Impacto Total Estimado', f"R$ {impacto['impacto_total']:,.2f}"]
        ]
        for periodo, valor in impacto['impacto_por_faixa'].items():
            consolidado_data.append([f'Impacto {periodo}', f"R$ {valor:,.2f}"])
        pd.DataFrame(consolidado_data, columns=['Métrica', 'Valor']).to_excel(writer, sheet_name='Consolidado', index=False)

        # --- ABA: Ligações por Dia da Semana ---
        ligacoes_dia_df = pd.DataFrame(list(consolidado['ligacoes_por_dia'].items()), columns=['Dia da Semana', 'Quantidade'])
        ligacoes_dia_df.to_excel(writer, sheet_name='Ligacoes_por_Dia', index=False)

        # --- ABA: Ligações por Hora do Dia (Pico) ---
        ligacoes_hora_df = pd.DataFrame(list(consolidado['horarios_pico'].items()), columns=['Horário (h)', 'Quantidade'])
        ligacoes_hora_df.to_excel(writer, sheet_name='Ligacoes_por_Hora', index=False)

        # --- ABA: Faixas de Ligações ---
        faixas_df = pd.DataFrame(list(faixas_ligacoes.items()), columns=['Faixa de Ligações', 'Quantidade de Telefones'])
        faixas_df.to_excel(writer, sheet_name='Faixas_Ligacoes', index=False)

        # --- ABA: Rechamadas - Resumo ---
        rechamadas_resumo_data = []
        for periodo, dados in consolidado['religacoes'].items():
            rechamadas_resumo_data.append([
                periodo,
                dados['quantidade'],
                dados['clientes_unicos'],
                f"{dados['tempo_medio_horas']:.1f}h",
                f"R$ {dados['impacto_financeiro']:,.2f}"
            ])
        pd.DataFrame(rechamadas_resumo_data, columns=['Período', 'Qtd. Rechamadas', 'Clientes Únicos', 'Tempo Médio', 'Impacto Financeiro']).to_excel(writer, sheet_name='Rechamadas_Resumo', index=False)

        # --- ABAS: Rechamadas - Detalhe por Período ---
        for periodo, dados in rechamadas_detalhe.items():
            if dados:
                df_rechamadas_detalhe = pd.DataFrame(dados)
                df_rechamadas_detalhe.to_excel(writer, sheet_name=f'Rechamadas_{periodo}', index=False)

        # --- ABA: Top 10 Clientes que Mais Ligaram ---
        top_10_clientes_df = pd.DataFrame(consolidado['top_clientes']).T.reset_index()
        top_10_clientes_df.columns = ['Telefone', 'Total Ligações', 'Frequência/Dia', 'Duração Total (min)']
        top_10_clientes_df.to_excel(writer, sheet_name='Top_10_Clientes', index=False)

        # --- ABA: Reincidência de Telefones (Dados do Gráfico) ---
        if not reincidentes_serie_filtrada.empty:
            reincidentes_serie_filtrada.rename("Quantidade_Ligacoes").to_frame().to_excel(writer, sheet_name='Reincidencia_Telefones', index=True)
        else:
            pd.DataFrame([["Nenhum telefone com mais de uma ligação para o filtro"]], columns=["Info"]).to_excel(writer, sheet_name='Reincidencia_Telefones', index=False)

        # --- ABA: Dados Processados Detalhados (Todos os Clientes) ---
        clientes_frequentes_todos.to_excel(writer, sheet_name='Dados_Clientes_Detalhados', index=True)

        # --- ABA: Contagem Bruta de Ligações por Telefone ---
        contagem_por_telefone_bruta.rename("Quantidade_Ligacoes").to_frame().to_excel(writer, sheet_name='Contagem_Bruta_Telefones', index=True)

    output.seek(0) # Volta para o início do buffer
    return output

# --- 5. FUNÇÃO PRINCIPAL (Streamlit App) ---
def streamlit_app():
    st.set_page_config(layout="wide", page_title="Análise de Rechamadas Call Center")

    st.title("📞 Análise de Rechamadas do Call Center")
    st.markdown("Faça o upload do seu arquivo de dados (CSV ou Excel com múltiplas abas) para analisar padrões de rechamadas e identificar clientes frequentes.")

    uploaded_file = st.file_uploader("Escolha um arquivo CSV ou Excel", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        df, error_message = analisar_ligacoes_callcenter_streamlit(uploaded_file)

        if error_message:
            st.error(error_message)
            st.stop() # Para a execução se houver erro no carregamento

        if df is not None and not df.empty:
            st.subheader("Dados Carregados (Amostra)")
            st.dataframe(df.head())

            st.markdown("---")
            st.subheader("Iniciando Análises...")

            # Executar as análises
            rechamadas_detalhe = identificar_faixas_rechamada(df)
            faixas_ligacoes, telefones_reincidentes, contagem_por_telefone_bruta = faixas_ligacoes_e_reincidentes(df)
            clientes_frequentes_todos = clientes_frequentes(df, N=df['telefone'].nunique()) # Pega todos os clientes para o Excel
            impacto_financeiro = calcular_impacto_financeiro(rechamadas_detalhe)
            consolidado = gerar_consolidado(df, rechamadas_detalhe, clientes_frequentes_todos, faixas_ligacoes, telefones_reincidentes, impacto_financeiro)
            reincidentes_serie_filtrada = contagem_por_telefone_bruta[contagem_por_telefone_bruta >= MIN_LIGACOES_GRAF]

            st.success("✅ Análises concluídas!")

            st.markdown("---")
            st.header("Sumário Executivo")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Ligações", f"{consolidado['estatisticas_gerais']['total_ligacoes']:,}")
                st.metric("Clientes Únicos", f"{consolidado['estatisticas_gerais']['clientes_unicos']:,}")
            with col2:
                st.metric("Números com >1 Ligação", f"{consolidado['telefones_reincidentes']:,}")
                st.metric("Média Ligações/Cliente", f"{consolidado['estatisticas_gerais']['media_ligacoes_por_cliente']:.2f}")
            with col3:
                st.metric("Duração Total (horas)", f"{consolidado['estatisticas_gerais']['duracao_total_horas']:,}h")
                st.metric("Duração Média/Ligação (min)", f"{consolidado['estatisticas_gerais']['duracao_media_minutos']:.1f} min")

            st.subheader("💰 Impacto Financeiro das Rechamadas")
            st.metric("Impacto Total Estimado", f"R$ {consolidado['impacto_financeiro']['impacto_total']:,.2f}")
            st.dataframe(pd.DataFrame(consolidado['impacto_financeiro']['impacto_por_faixa'].items(), columns=['Período', 'Impacto Financeiro (R$)']).set_index('Período'))

            st.markdown("---")
            st.header("Visualizações Detalhadas")

            # Gerar e exibir os gráficos
            dashboard_plots = create_dashboard_plots(consolidado, reincidentes_serie_filtrada, MIN_LIGACOES_GRAF)

            st.subheader("📅 Ligações por Dia da Semana")
            st.pyplot(dashboard_plots['ligacoes_por_dia'])

            st.subheader("🕐 Top 5 Horários de Pico")
            st.pyplot(dashboard_plots['horarios_pico'])

            st.subheader("📊 Distribuição por Faixas de Ligações")
            st.pyplot(dashboard_plots['faixas_ligacoes'])

            st.subheader("📞 Rechamadas por Período")
            st.pyplot(dashboard_plots['rechamadas_por_periodo'])

            st.subheader(f"📉 Reincidência (Telefones com ≥{MIN_LIGACOES_GRAF} Ligações)")
            st.pyplot(dashboard_plots['hist_reincidencia'])

            st.subheader("🏆 Top 10 Clientes que Mais Ligaram")
            st.pyplot(dashboard_plots['top_clientes'])

            st.markdown("---")
            st.header("Download dos Resultados")

            excel_buffer = to_excel_buffer(df, rechamadas_detalhe, clientes_frequentes_todos, consolidado, faixas_ligacoes, contagem_por_telefone_bruta, reincidentes_serie_filtrada)
            st.download_button(
                label="📥 Baixar Relatório Completo em Excel",
                data=excel_buffer,
                file_name=f"analise_callcenter_completa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("Nenhum dado válido para análise após o processamento.")

if __name__ == "__main__":
    streamlit_app()
