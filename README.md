================================================================================
   PROSPORTS AI v2 — SISTEMA DE SCRAPING Y PREDICCION DEPORTIVA AVANZADA
================================================================================

Sistema integral en Python para extraer datos deportivos en tiempo real desde
Sofascore y Flashscore, unificar nombres de equipos entre plataformas y generar
pronosticos matematicos en 8 mercados (incluyendo Jugadores / Player Props).


--------------------------------------------------------------------------------
1. COMO INICIAR EL SISTEMA RAPIDAMENTE
--------------------------------------------------------------------------------

  OPCION A: Con doble clic (Recomendado en Windows)
  -------------------------------------------------
  Haz doble clic sobre el archivo:
      iniciar_dashboard.bat

  Esto abrira automaticamente el servidor web en:
      http://localhost:8501


  OPCION B: Desde la terminal (PowerShell / CMD)
  ----------------------------------------------
  Para abrir el Dashboard Web Interactivo:
      .\.venv\Scripts\streamlit.exe run src/dashboard.py --server.port 8501

  Para ejecutar el analisis por consola (CLI):
      .\.venv\Scripts\python.exe -m src.main --date 2026-11-04 --limit 8


--------------------------------------------------------------------------------
2. ARQUITECTURA Y ESTRUCTURA DE ARCHIVOS
--------------------------------------------------------------------------------

  scrapin/
  │
  ├── iniciar_dashboard.bat        -> Lanzador directo con doble clic
  ├── DOCUMENTACION.html           -> Guia visual interactiva (abrir en navegador)
  ├── requirements.txt             -> Dependencias del entorno virtual (.venv)
  │
  ├── data/
  │   ├── sports_analytics.db      -> Base de datos SQLite (partidos, equipos, alias)
  │   └── cache/                   -> Cache local JSON (evita bloqueos de IP)
  │
  └── src/
      ├── config.py                -> Ligas prioritarias, rutas y parametros
      ├── main.py                  -> Interfaz de linea de comandos (CLI)
      ├── dashboard.py             -> Aplicacion Web Interactiva (Streamlit v2)
      │
      ├── scrapers/
      │   ├── base.py              -> Cliente TLS (curl_cffi firefox133) + Cache
      │   ├── sofascore.py         -> Extractor de Sofascore (xG, cuotas, arbitro,
      │   │                           alineaciones, lesionados y jugadores)
      │   └── flashscore.py        -> Decodificador del feed (~AA÷...¬) Flashscore
      │
      ├── models/
      │   ├── schemas.py           -> Esquemas de validacion (Pydantic)
      │   └── database.py          -> Tablas SQLite (SQLAlchemy ORM)
      │
      └── analytics/
          ├── entity_resolver.py   -> Unificador difuso de equipos (RapidFuzz)
          └── predictor.py         -> Motor matematico (Poisson + Dixon-Coles +
                                      Elo + Local/Visita + Arbitro + Kelly)


--------------------------------------------------------------------------------
3. COMO FUNCIONA EL MOTOR DE PREDICCION (PASO A PASO)
--------------------------------------------------------------------------------

  [PASO 1] Extraccion Sigilosa (Bypass Antibot)
           Usa 'curl_cffi' emulando la huella TLS de Firefox 133 para consultar
           la API interna de Sofascore y los feeds de Flashscore sin bloqueos
           de Cloudflare y con cache en disco.

  [PASO 2] Ponderacion Local vs. Visitante + Decaimiento Exponencial
           Analiza los ultimos partidos de cada club dando un +40% de peso a
           los encuentros jugados en la misma condicion (Local en casa vs.
           Visitante fuera) y mayor peso a los partidos mas recientes.

  [PASO 3] Modelo Hibrido: xG Real + Goles + Rating Elo + H2H
           Combina el xG (Expected Goals) real creado y concedido con el
           diferencial de Rating Elo dinamico (incluyendo posicion en tabla)
           y el historial de enfrentamientos directos (H2H).

  [PASO 4] Penalizacion Automatica por Bajas Clave
           Cruza la lista de lesionados/suspendidos del partido contra los
           lideres de remates al arco y xG del equipo. Si falta una figura
           ofensiva, descuenta automaticamente su porcentaje de ataque.

  [PASO 5] Calibracion de Tarjetas con el Arbitro Asignado
           Extrae el arbitro oficial del partido y su promedio historico de
           tarjetas amarillas/rojas por encuentro, ponderando 55% equipos y
           45% rigor del arbitro.

  [PASO 6] Matriz Exacta de Dixon-Coles y 8 Mercados
           Calcula las probabilidades exactas para:
             1. Resultado Final (1X2) y Doble Oportunidad (1X / X2)
             2. Goles Tiempo Completo (Over/Under 1.5, 2.5 y 3.5)
             3. Ambos Equipos Anotan (BTTS Si / No)
             4. Goles y Corners en el 1er Tiempo (1H: +0.5 1H, +4.5 Corners 1H)
             5. Remates al Arco (+7.5, +8.5, +9.5 Shots on Target)
             6. Remates Totales (+22.5, +24.5, +26.5 Total Shots)
             7. Corners Totales (+8.5, +9.5, +10.5 Corners)
             8. Tarjetas Amarillas (+3.5, +4.5, +5.5 Tarjetas)

  [PASO 7] Probabilidades por Jugador (Player Props)
           Calcula mediante distribucion de Poisson individual que jugadores
           tienen mayor probabilidad de realizar 1+ o 2+ remates al arco,
           2+ remates totales o anotar gol en cualquier momento.

  [PASO 8] Value Bets, Smart Money, Criterio de Kelly y Backtesting
           - Compara las probabilidades del modelo contra las cuotas de las
             casas de apuestas para detectar Value Bets.
           - Compara cuota de apertura vs cuota actual (Dropping Odds).
           - Calcula el stake recomendado con Kelly Fraccional (1/4).
           - En partidos finalizados, verifica automaticamente cada mercado
             con ACIERTO (OK) o FALLO (X) para medir la efectividad real.

================================================================================
NOTA: Si deseas ver la documentacion con graficos y tarjetas de colores en tu
navegador, abre con doble clic el archivo 'DOCUMENTACION.html'.
================================================================================
