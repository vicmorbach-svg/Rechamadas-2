import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import io

warnings.filterwarnings('ignore')

# --- CONFIGURAÇÕES GLOBAIS ---
VALOR_LIGACAO = 7.56
MIN_LIGACOES_GRAF = 2

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
        else:
            return int(float(duracao_str))
    except ValueError:
        return 0

def detectar_coluna_sugerida(df, tipo='datetime'):
    """
    Detecta e sugere colunas baseado no tipo solicitado.
    tipo: 'datetime', 'telefone', 'duracao'
    """
    sugestoes = []

    if tipo == 'datetime':
        palavras_chave = ['data', 'datetime', 'timestamp', 'hora', 'time', 'date']
        for col in df.columns:
            col_lower = col.lower()
            if any(palavra in col_lower for palavra in palavras_chave):
                sugestoes.append(col)

    elif tipo == 'telefone':
        palavras_chave = ['telefone', 'phone', 'numero', 'fone', 'tel', 'ani', 'cliente', 'customer']
        for col in df.columns:
            col_lower = col.lower()
            if any(palavra in col_lower for palavra in palavras_chave):
                sugestoes.append(col)

    elif tipo == 'duracao':
        palavras_chave = ['duracao', 'duração', 'duration', 'tempo', 'time']
        for col in df.columns:
            col_lower = col.lower()
            if any(palavra in col_lower for palavra in palavras_chave):
                sugestoes.append(col)

    return sugestoes[0] if sugestoes else None

# --- 2. CARREGAMENTO INICIAL DOS DADOS ---
@st.cache_data(show_spinner="Carregando arquivo...")
def carregar_arquivo_inicial(uploaded_file):
    """
    Carrega o arquivo e retorna DataFrame bruto para seleção de colunas.
    """
    if uploaded_file is None:
        return None, "Nenhum arquivo carregado."

    file_details = {
        "filename": uploaded_file.name, 
        "filetype": uploaded_file.type, 
        "filesize": uploaded_file.size
    }

    st.write(f"📖 Arquivo: **{file_details['filename']}** ({file_details['filesize'] / 1024:.2f} KB)")

    dfs = []
    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_extension == 'csv':
            uploaded_file_content = uploaded_file.getvalue()

            for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                for sep in [',', ';', '\t']:
                    try:
                        df_temp = pd.read_csv(
                            io.StringIO(uploaded_file_content.decode(encoding)), 
                            sep=sep
                        )
                        if not df_temp.empty and len(df_temp.columns) > 1:
                            st.success(f"✅ CSV carregado com encoding: {encoding}, separador: '{sep}'")
                            dfs.append(df_temp)
                            break
                    except Exception:
                        continue
                if dfs:
                    break

        elif file_extension in ['xlsx', 'xls']:
            uploaded_file.seek(0)
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            st.info(f"📊 Excel com {len(sheet_names)} aba(s): {', '.join(sheet_names)}")

            for sheet_name in sheet_names:
                df_temp = pd.read_excel(excel_file, sheet_name=sheet_name)
                if not df_temp.empty and len(df_temp.columns) > 1:
                    dfs.append(df_temp)
                    st.success(f"✅ Aba '{sheet_name}' carregada")

        if not dfs:
            return None, "Nenhum dado válido encontrado no arquivo."

        df_completo = pd.concat(dfs, ignore_index=True)
        df_completo = df_completo.loc[:, ~df_completo.columns.str.contains('^Unnamed', na=False)]

        st.success(f"✅ Total de registros carregados: **{len(df_completo):,}**")
        return df_completo, None

    except Exception as e:
        return None, f"Erro ao carregar arquivo: {str(e)}"

# --- 3. PROCESSAMENTO DOS DADOS COM COLUNAS SELECIONADAS ---
@st.cache_data(show_spinner="Processando dados...")
def processar_dados(df_bruto, col_datetime, col_cliente, col_duracao=None):
    """
    Processa o DataFrame com as colunas selecionadas pelo usuário.
    """
    df = df_bruto.copy()

    # Renomear colunas selecionadas
    df = df.rename(columns={
        col_datetime: 'data_hora_original',
        col_cliente: 'telefone_original'
    })

    if col_duracao and col_duracao in df_bruto.columns:
        df = df.rename(columns={col_duracao: 'duracao_original'})

    # --- PROCESSAR DATA/HORA ---
    st.write("🔄 Processando coluna de data/hora...")

    # Limpeza agressiva da coluna de data/hora
    df['data_hora_limpa'] = df['data_hora_original'].astype(str).str.strip()

    # Remover espaços múltiplos, tabs, quebras de linha
    df['data_hora_limpa'] = df['data_hora_limpa'].str.replace(r'\s+', ' ', regex=True)

    # Remover caracteres invisíveis comuns (zero-width space, etc)
    df['data_hora_limpa'] = df['data_hora_limpa'].str.replace(r'[\u200b\u200c\u200d\ufeff]', '', regex=True)

    # Remover espaços ao redor do T (se existir)
    df['data_hora_limpa'] = df['data_hora_limpa'].str.replace(r'\s*T\s*', 'T', regex=True)

    # Mostrar amostra dos dados limpos para debug
    st.write("**Amostra dos dados de data/hora após limpeza:**")
    amostra_datas = df['data_hora_limpa'].head(10).tolist()
    for i, data in enumerate(amostra_datas, 1):
        st.text(f"{i}. '{data}' (tipo: {type(data).__name__}, len: {len(str(data))})")

    # Formatos de data/hora em ordem de prioridade
    # IMPORTANTE: Formatos ISO com T devem vir PRIMEIRO
    datetime_formats = [
        '%Y-%m-%dT%H:%M:%S',      # ISO 8601 com T (SEU FORMATO) - PRIORIDADE MÁXIMA
        '%Y-%m-%dT%H:%M:%S.%f',   # ISO com T e microsegundos
        '%Y-%m-%dT%H:%M',         # ISO com T sem segundos
        '%Y-%m-%d %H:%M:%S',      # ISO com espaço
        '%Y-%m-%d %H:%M:%S.%f',   # ISO com espaço e microsegundos
        '%Y-%m-%d %H:%M',         # ISO com espaço sem segundos
        '%d/%m/%Y %H:%M:%S',      # BR formato completo
        '%d/%m/%Y %H:%M',         # BR sem segundos
        '%d-%m-%Y %H:%M:%S',      # BR com traço
        '%d-%m-%Y %H:%M',         # BR com traço sem segundos
        '%Y/%m/%d %H:%M:%S',      # ISO com barra
        '%Y/%m/%d %H:%M',         # ISO com barra sem segundos
        '%d/%m/%Y',               # BR só data
        '%Y-%m-%d',               # ISO só data
        '%Y/%m/%d',               # ISO com barra só data
    ]

    # Inicializar coluna de datetime
    df['datetime'] = pd.NaT
    total_rows = len(df)
    converted_count = 0

    st.write(f"\n📊 Total de registros a converter: **{total_rows:,}**")
    st.write("🔍 Tentando conversão com diferentes formatos...\n")

    # Tentar cada formato sequencialmente
    for fmt in datetime_formats:
        # Contar quantos ainda estão como NaT
        mask = df['datetime'].isna()
        pendentes = mask.sum()

        if pendentes == 0:
            st.success(f"✅ Todos os {total_rows:,} registros foram convertidos!")
            break

        # Tentar converter apenas os que ainda são NaT
        try:
            df.loc[mask, 'datetime'] = pd.to_datetime(
                df.loc[mask, 'data_hora_limpa'], 
                format=fmt, 
                errors='coerce'
            )

            # Contar quantos foram convertidos nesta iteração
            newly_converted = (~df.loc[mask, 'datetime'].isna()).sum()

            if newly_converted > 0:
                converted_count += newly_converted
                percentual = (converted_count / total_rows) * 100
                st.info(f"   ✓ Formato `{fmt}`: converteu **{newly_converted:,}** registros | "
                       f"Total: **{converted_count:,}/{total_rows:,}** ({percentual:.1f}%)")

        except Exception as e:
            st.warning(f"   ⚠️ Erro ao tentar formato `{fmt}`: {str(e)}")
            continue

    # Última tentativa: inferência automática para os que restaram
    mask_final = df['datetime'].isna()
    pendentes_final = mask_final.sum()

    if pendentes_final > 0:
        st.warning(f"\n⚠️ Ainda há **{pendentes_final:,}** datas não convertidas. "
                  f"Tentando inferência automática...")

        # Mostrar exemplos dos que falharam
        st.write("**Exemplos de datas que falharam:**")
        exemplos_falha = df.loc[mask_final, 'data_hora_limpa'].head(10).tolist()
        for i, data in enumerate(exemplos_falha, 1):
            st.text(f"{i}. '{data}'")

        try:
            # Tentar com dayfirst=True
            df.loc[mask_final, 'datetime'] = pd.to_datetime(
                df.loc[mask_final, 'data_hora_limpa'], 
                dayfirst=True, 
                errors='coerce'
            )

            final_converted = (~df.loc[mask_final, 'datetime'].isna()).sum()

            if final_converted > 0:
                converted_count += final_converted
                percentual = (converted_count / total_rows) * 100
                st.info(f"   ✓ Inferência automática: converteu **{final_converted:,}** registros | "
                       f"Total: **{converted_count:,}/{total_rows:,}** ({percentual:.1f}%)")

        except Exception as e:
            st.error(f"   ❌ Erro na inferência automática: {str(e)}")

    # Verificar quantos ainda são NaT após todas as tentativas
    registros_antes = len(df)
    nulos_finais = df['datetime'].isna().sum()

    if nulos_finais > 0:
        st.error(f"\n🚨 **{nulos_finais:,}** registros ({(nulos_finais/total_rows)*100:.1f}%) "
                f"não puderam ser convertidos e serão removidos.")

        # Mostrar mais exemplos dos que falharam
        st.write("**Últimos 20 exemplos de datas que falharam completamente:**")
        mask_nulos = df['datetime'].isna()
        exemplos_nulos = df.loc[mask_nulos, ['data_hora_original', 'data_hora_limpa']].head(20)
        st.dataframe(exemplos_nulos)

        # Remover registros com datetime inválido
        df = df.dropna(subset=['datetime'])

        st.warning(f"⚠️ Removidos {registros_antes - len(df):,} registros com data/hora inválida.")
    else:
        st.success(f"\n🎉 **100% dos registros convertidos com sucesso!** ({converted_count:,}/{total_rows:,})")

    if df.empty:
        st.error("🚨 Todos os registros foram removidos devido a datas/horas inválidas. "
                "Verifique o formato da coluna de data/hora.")
        return None

    # Mostrar estatísticas da conversão
    st.write("\n📈 **Estatísticas da conversão:**")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Registros Originais", f"{total_rows:,}")
    with col2:
        st.metric("Convertidos com Sucesso", f"{converted_count:,}")
    with col3:
        taxa_sucesso = (converted_count / total_rows) * 100
        st.metric("Taxa de Sucesso", f"{taxa_sucesso:.1f}%")

    # Mostrar range de datas
    st.write(f"\n📅 **Período dos dados:** {df['datetime'].min():%d/%m/%Y %H:%M:%S} "
            f"até {df['datetime'].max():%d/%m/%Y %H:%M:%S}")

    # --- PROCESSAR TELEFONE/CLIENTE ---
    st.write("\n🔄 Processando coluna de cliente/telefone...")
    df['telefone'] = df['telefone_original'].astype(str).str.replace(r'[^\d]', '', regex=True)

    registros_antes = len(df)
    df = df[df['telefone'].str.len() >= 8]

    if len(df) < registros_antes:
        st.warning(f"⚠️ Removidos {registros_antes - len(df):,} registros com telefone inválido "
                  f"(menos de 8 dígitos)")

    # --- PROCESSAR DURAÇÃO ---
    if 'duracao_original' in df.columns:
        st.write("🔄 Processando coluna de duração...")
        df['duracao_segundos'] = df['duracao_original'].apply(converter_duracao_para_segundos)
    else:
        df['duracao_segundos'] = 0
        st.info("ℹ️ Sem coluna de duração - usando valor padrão 0")

    # Ordenar para análise
    df = df.sort_values(['telefone', 'datetime']).reset_index(drop=True)

    st.success(f"\n✅ **{len(df):,}** ligações processadas com sucesso!")

    return df


# --- 4. FUNÇÕES DE ANÁLISE (mantidas como no original) ---
def identificar_faixas_rechamada(df):
    """Identifica rechamadas em faixas de 0-24h, 24-48h, 48-72h."""
    rechamadas = {'0-24h': [], '24-48h': [], '48-72h': []}
    for telefone, grupo in df.groupby('telefone'):
        if len(grupo) < 2:
            continue
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
            if diff_h <= 24:
                rechamadas['0-24h'].append(rec)
            elif 24 < diff_h <= 48:
                rechamadas['24-48h'].append(rec)
            elif 48 < diff_h <= 72:
                rechamadas['48-72h'].append(rec)
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
        lambda x: x['total_ligacoes'] / max(1, x['periodo_atividade_dias']) if x['periodo_atividade_dias'] > 0 else x['total_ligacoes'], 
        axis=1
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
    dias_pt = {
        'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
        'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
    }
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

# --- 5. FUNÇÕES DE VISUALIZAÇÃO (mantidas como no original) ---
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
    for bar in bars_dia:
        ax_dia.text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                   f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=8)
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
    for bar in bars_hora:
        ax_hora.text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                    f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plots['horarios_pico'] = fig_hora

    # 3. Faixas de Ligações
    fig_faixas, ax_faixas = plt.subplots(figsize=(8, 8))
    faixas_labels = [k for k, v in consolidado['faixas_ligacoes'].items() if v > 0]
    faixas_sizes = [v for v in consolidado['faixas_ligacoes'].values() if v > 0]
    if faixas_sizes:
        colors = plt.cm.Pastel1(np.linspace(0, 1, len(faixas_sizes)))
        ax_faixas.pie(faixas_sizes, labels=faixas_labels, autopct='%1.1f%%', 
                     colors=colors, startangle=90, textprops={'fontsize': 10})
        ax_faixas.set_title('📊 Distribuição por Faixas de Ligações', fontsize=12)
    else:
        ax_faixas.text(0.5, 0.5, 'Nenhuma faixa de ligação para exibir', 
                      horizontalalignment='center', verticalalignment='center', 
                      transform=ax_faixas.transAxes, fontsize=10)
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
    for bar in bars_relig:
        ax_relig.text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                     f'{int(bar.get_height()):,}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plots['rechamadas_por_periodo'] = fig_relig

    # 5. Histograma de Reincidência
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
    if not reincidentes_serie.empty:
        sns.histplot(reincidentes_serie, bins=range(min_ligacoes_graf, reincidentes_serie.max() + 2), 
                    kde=False, ax=ax_hist, color='darkorchid')
        ax_hist.set_title(f'📉 Reincidência (Telefones com ≥{min_ligacoes_graf} Ligações)', fontsize=12)
        ax_hist.set_xlabel('Número de Ligações', fontsize=10)
        ax_hist.set_ylabel('Quantidade de Telefones', fontsize=10)
        ax_hist.tick_params(axis='x', labelsize=9)
        ax_hist.tick_params(axis='y', labelsize=9)
    else:
        ax_hist.text(0.5, 0.5, 'Nenhum telefone com ≥2 ligações', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax_hist.transAxes, fontsize=10)
        ax_hist.axis('off')
    plt.tight_layout()
    plots['hist_reincidencia'] = fig_hist

    # 6. Top 10 Clientes
    fig_top_clientes, ax_top_clientes = plt.subplots(figsize=(8, 4))
    top_clientes_df = pd.DataFrame(consolidado['top_clientes']).T
    if not top_clientes_df.empty:
        bars_top = ax_top_clientes.bar(top_clientes_df.index.astype(str), 
                                       top_clientes_df['total_ligacoes'], 
                                       color='teal', edgecolor='black')
        ax_top_clientes.set_title('🏆 Top 10 Clientes que Mais Ligaram', fontsize=12)
        ax_top_clientes.set_xlabel('Telefone', fontsize=10)
        ax_top_clientes.set_ylabel('Total de Ligações', fontsize=10)
        ax_top_clientes.tick_params(axis='x', labelsize=9, rotation=45)
        ax_top_clientes.set_xticklabels(top_clientes_df.index.astype(str), rotation=45, ha='right', fontsize=9)
        ax_top_clientes.tick_params(axis='y', labelsize=9)
        for bar in bars_top:
            ax_top_clientes.text(bar.get_x() + bar.get_width()/2., bar.get_height(), 
                                f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=7)
    else:
        ax_top_clientes.text(0.5, 0.5, 'Nenhum cliente frequente encontrado', 
                           horizontalalignment='center', verticalalignment='center', 
                           transform=ax_top_clientes.transAxes, fontsize=10)
        ax_top_clientes.axis('off')
    plt.tight_layout()
    plots['top_clientes'] = fig_top_clientes

    return plots

def to_excel_buffer(df, rechamadas_detalhe, clientes_frequentes_todos, consolidado, 
                   faixas_ligacoes, contagem_por_telefone_bruta, reincidentes_serie_filtrada):
    """Salva todos os resultados em Excel."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Consolidado Geral
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
            ['', ''],
            ['IMPACTO FINANCEIRO', ''],
            ['Total de Rechamadas', impacto['total_religacoes']],
            ['Custo por Ligação', f"R$ {impacto['valor_por_ligacao']:.2f}"],
            ['Impacto Total Estimado', f"R$ {impacto['impacto_total']:,.2f}"]
        ]
        for periodo, valor in impacto['impacto_por_faixa'].items():
            consolidado_data.append([f'Impacto {periodo}', f"R$ {valor:,.2f}"])
        pd.DataFrame(consolidado_data, columns=['Métrica', 'Valor']).to_excel(writer, sheet_name='Consolidado', index=False)

        # Outras abas
        pd.DataFrame(list(consolidado['ligacoes_por_dia'].items()), 
                    columns=['Dia da Semana', 'Quantidade']).to_excel(writer, sheet_name='Ligacoes_por_Dia', index=False)

        pd.DataFrame(list(consolidado['horarios_pico'].items()), 
                    columns=['Horário (h)', 'Quantidade']).to_excel(writer, sheet_name='Ligacoes_por_Hora', index=False)

        pd.DataFrame(list(faixas_ligacoes.items()), 
                    columns=['Faixa de Ligações', 'Quantidade de Telefones']).to_excel(writer, sheet_name='Faixas_Ligacoes', index=False)

        # Rechamadas
        rechamadas_resumo_data = []
        for periodo, dados in consolidado['religacoes'].items():
            rechamadas_resumo_data.append([
                periodo,
                dados['quantidade'],
                dados['clientes_unicos'],
                f"{dados['tempo_medio_horas']:.1f}h",
                f"R$ {dados['impacto_financeiro']:,.2f}"
            ])
        pd.DataFrame(rechamadas_resumo_data, 
                    columns=['Período', 'Qtd. Rechamadas', 'Clientes Únicos', 'Tempo Médio', 'Impacto Financeiro']
                    ).to_excel(writer, sheet_name='Rechamadas_Resumo', index=False)

        for periodo, dados in rechamadas_detalhe.items():
            if dados:
                pd.DataFrame(dados).to_excel(writer, sheet_name=f'Rechamadas_{periodo}', index=False)

        # Top clientes
        top_10_clientes_df = pd.DataFrame(consolidado['top_clientes']).T.reset_index()
        top_10_clientes_df.columns = ['Telefone', 'Total Ligações', 'Frequência/Dia', 'Duração Total (min)']
        top_10_clientes_df.to_excel(writer, sheet_name='Top_10_Clientes', index=False)

        # Reincidência
        if not reincidentes_serie_filtrada.empty:
            reincidentes_serie_filtrada.rename("Quantidade_Ligacoes").to_frame().to_excel(
                writer, sheet_name='Reincidencia_Telefones', index=True)

        # Dados detalhados
        clientes_frequentes_todos.to_excel(writer, sheet_name='Dados_Clientes_Detalhados', index=True)
        contagem_por_telefone_bruta.rename("Quantidade_Ligacoes").to_frame().to_excel(
            writer, sheet_name='Contagem_Bruta_Telefones', index=True)

    output.seek(0)
    return output

# --- 6. APLICAÇÃO STREAMLIT PRINCIPAL ---
def streamlit_app():
    st.set_page_config(layout="wide", page_title="Análise de Rechamadas Call Center")

    st.title("📞 Análise de Rechamadas do Call Center")
    st.markdown("Faça o upload do seu arquivo de dados (CSV ou Excel) para analisar padrões de rechamadas.")

    # Upload do arquivo
    uploaded_file = st.file_uploader("Escolha um arquivo CSV ou Excel", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        # Carregar arquivo inicial
        df_bruto, error_message = carregar_arquivo_inicial(uploaded_file)

        if error_message:
            st.error(error_message)
            st.stop()

        if df_bruto is not None and not df_bruto.empty:
            st.markdown("---")
            st.subheader("📋 Seleção de Colunas")

            # Mostrar amostra dos dados
            with st.expander("👁️ Visualizar amostra dos dados carregados"):
                st.dataframe(df_bruto.head(10))

            # Detectar sugestões automáticas
            sugestao_datetime = detectar_coluna_sugerida(df_bruto, 'datetime')
            sugestao_telefone = detectar_coluna_sugerida(df_bruto, 'telefone')
            sugestao_duracao = detectar_coluna_sugerida(df_bruto, 'duracao')

            colunas_disponiveis = list(df_bruto.columns)

            # Interface de seleção de colunas
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**🕐 Coluna de Data/Hora** *(obrigatória)*")
                idx_datetime = colunas_disponiveis.index(sugestao_datetime) if sugestao_datetime else 0
                col_datetime = st.selectbox(
                    "Selecione a coluna que contém data/hora:",
                    colunas_disponiveis,
                    index=idx_datetime,
                    key='datetime_col'
                )
                if sugestao_datetime:
                    st.caption(f"✅ Sugestão automática: {sugestao_datetime}")

            with col2:
                st.markdown("**📱 Coluna de Cliente/Telefone** *(obrigatória)*")
                idx_telefone = colunas_disponiveis.index(sugestao_telefone) if sugestao_telefone else 0
                col_cliente = st.selectbox(
                    "Selecione a coluna que identifica o cliente:",
                    colunas_disponiveis,
                    index=idx_telefone,
                    key='cliente_col'
                )
                if sugestao_telefone:
                    st.caption(f"✅ Sugestão automática: {sugestao_telefone}")

            with col3:
                st.markdown("**⏱️ Coluna de Duração** *(opcional)*")
                opcoes_duracao = ['Nenhuma'] + colunas_disponiveis
                idx_duracao = opcoes_duracao.index(sugestao_duracao) if sugestao_duracao else 0
                col_duracao = st.selectbox(
                    "Selecione a coluna de duração (se houver):",
                    opcoes_duracao,
                    index=idx_duracao,
                    key='duracao_col'
                )
                if sugestao_duracao:
                    st.caption(f"✅ Sugestão automática: {sugestao_duracao}")

            # Validação
            if col_datetime == col_cliente:
                st.error("❌ As colunas de data/hora e cliente não podem ser iguais!")
                st.stop()

            # Botão para processar
            if st.button("🚀 Processar Dados", type="primary"):
                col_duracao_final = None if col_duracao == 'Nenhuma' else col_duracao

                # Processar dados
                df_processado = processar_dados(df_bruto, col_datetime, col_cliente, col_duracao_final)

                if df_processado is None or df_processado.empty:
                    st.error("Não foi possível processar os dados. Verifique as colunas selecionadas.")
                    st.stop()

                # Armazenar no session_state para persistir
                st.session_state['df_processado'] = df_processado
                st.session_state['processamento_concluido'] = True

            # Se já processou, mostrar análises
            if st.session_state.get('processamento_concluido', False):
                df = st.session_state['df_processado']

                st.markdown("---")
                st.subheader("📊 Executando Análises...")

                # Análises
                rechamadas_detalhe = identificar_faixas_rechamada(df)
                faixas_ligacoes, telefones_reincidentes, contagem_por_telefone_bruta = faixas_ligacoes_e_reincidentes(df)
                clientes_frequentes_todos = clientes_frequentes(df, N=df['telefone'].nunique())
                impacto_financeiro = calcular_impacto_financeiro(rechamadas_detalhe)
                consolidado = gerar_consolidado(df, rechamadas_detalhe, clientes_frequentes_todos, 
                                               faixas_ligacoes, telefones_reincidentes, impacto_financeiro)
                reincidentes_serie_filtrada = contagem_por_telefone_bruta[contagem_por_telefone_bruta >= MIN_LIGACOES_GRAF]

                st.success("✅ Análises concluídas!")

                # Sumário Executivo
                st.markdown("---")
                st.header("📈 Sumário Executivo")

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
                st.dataframe(pd.DataFrame(consolidado['impacto_financeiro']['impacto_por_faixa'].items(), 
                                        columns=['Período', 'Impacto Financeiro (R$)']).set_index('Período'))

                # Visualizações
                st.markdown("---")
                st.header("📊 Visualizações Detalhadas")

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

                # Download
                st.markdown("---")
                st.header("💾 Download dos Resultados")

                excel_buffer = to_excel_buffer(df, rechamadas_detalhe, clientes_frequentes_todos, 
                                              consolidado, faixas_ligacoes, contagem_por_telefone_bruta, 
                                              reincidentes_serie_filtrada)
                st.download_button(
                    label="📥 Baixar Relatório Completo em Excel",
                    data=excel_buffer,
                    file_name=f"analise_callcenter_completa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    streamlit_app()
