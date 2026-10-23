Kwik Fit Mobile — Resource & Demand Planning Assessment
Python analysis · Power BI reporting · operational recommendations
Review period: 8 October–22 December 2024 (76 calendar days, inclusive)
This project analyses three operational datasets—Service Appointments, Territory Assignments and Technicians—for the Junior Resource & Demand Planning Analyst assessment. It examines where service demand occurs, how that demand translates into planned workload, how technicians cover customer territories, and which service disruptions warrant action.
Assessment deliverables: a three-slide management presentation, a two-page Power BI report and supporting Python analysis. The presentation and the final Power BI report may be supplied separately from this code archive; the outputs/powerbi/KwikFit.pbix file inside older archives is not necessarily the final dashboard.
Questions investigated
1. Demand: Where and when are appointments booked, and does ranking territories by appointment count tell the same story as ranking them by planned service hours?
2. Resources and coverage: How is observed technician activity distributed, and which customer territories are served by technicians based elsewhere?
3. Service disruption: What failure reasons affect appointments, and where is booked workload exposed to internal service failures?
4. Planning action: Can descriptive patterns, simple predictive baselines and operational checks help prioritise practical interventions?
Three headline findings
Finding	Evidence	Planning implication
Appointment volume does not equal workload.	Leicester ranks 23rd by appointments but 4th by planned hours: 791 appointments and approximately 1,184 planned hours.	Compare both volume and planned duration when reviewing territory workload and allocating support.
Coverage crosses home-territory boundaries.	10,515 of 47,411 appointments with known home/customer territory assignments (22.2%) were served outside the technician's home territory.	Review territory-level dependence on visiting technicians alongside actual shifts, availability and journey feasibility before changing assignments.
Stock-related disruptions have different recorded-start patterns.	A recorded Actual Start appears on 1,320/1,390 (95.0%) wrong-parts failures, 293/343 (85.4%) locking-wheel-nut failures, but 160/1,182 (13.5%) tyre-stock failures.	Pilot pre-dispatch checks of parts compatibility, stock availability and locking-wheel-nut readiness. Measure outcomes before scaling.


Recommended pilot: For four weeks, select comparable teams/territories for pre-dispatch parts-readiness checks. Log checks performed, issues caught before the appointment, additional checking time, failure and reschedule outcomes, and any verified technician time released. Compare against a suitable baseline or control group; review with operations and the planning manager before rollout.
Other relevant observations
- 47,747 operational appointments; 29,882.05 total planned service hours; 9,208 classified failures (19.28%).
- 199 technicians in the roster; 158 with at least one recorded operational appointment; 6,649 distinct observed technician–appointment days. Technicians without appointments in this extract are not necessarily idle or unavailable.
- 5,128 failures classified as internal (Stock, Resource or Booking admin); 2,370 have a recorded Actual Start, associated with 1,426.3 booked planned hours. This is exposure, not confirmed time lost or recoverable capacity.
- 336 appointments lack a customer territory; 468 have no mapped region, including 132 with special suspended-territory labels. 616 have no lead-time value; those records are concentrated in Inspect & Fit work.
- The optional subsequent-same-vehicle-appointment analysis observes 4,557/9,193 (49.6%) among failed appointments meeting its identifier/date-window eligibility rules. It does not establish confirmed rework or full follow-up coverage.
Predictive model, simply explained
src/05_model.py fits a logistic regression to estimate the probability that an appointment will be marked failed, using selected scheduling, booking, work-type and regional fields. Numeric values are imputed/scaled and categories are encoded; the split is time-based (earlier appointments for training, December for testing) rather than a random shuffle.
- December holdout ROC AUC: 0.580 for any failure: limited ability to distinguish failed from non-failed appointments.
- Top 10% of model-ranked December appointments: 33.0% failed versus a 21.3% overall December failure rate (approximately 1.55× lift). The ranking is across the whole test period, not a simulated daily planner queue.
- A stock-failure variant scored 0.603 AUC; wrong-parts-ordered variant scored 0.531 AUC.
Interpretation: This is a transparent exploratory baseline, not a deployment-ready scheduling or AI automation system. Weak discrimination with these available features supports testing targeted operational checks instead of claiming reliable individual risk predictions. Historical feature availability, calibration, drift and intervention costs would need further evaluation.
A separate src/09_forecast.py provides an exploratory day-of-week demand baseline, not a validated workforce-capacity forecast.
Method and judgement
1. Convert source extracts to parquet and standardise Excel date serials/numeric fields (00_convert.py).
2. Exclude administrative placeholders and restrict operational analysis to 8 Oct–22 Dec 2024, trimming extract-boundary effects; create an auditable effective appointment date (01_clean.py).
3. Join technician home territory and customer territory; retain missing assignments as unknown, not automatically cross-territory (01_clean.py, 03_capacity.py).
4. Measure demand in both appointments and planned service hours, then investigate regional/territory work mix and observed activity (02_demand.py, 03_capacity.py).
5. Analyse failures, lead time, cancellations, recorded starts and booked-hour exposure; develop a measurable operational pilot (04_opportunity_loss.py, 06_recommendation.py).
6. Evaluate supplementary modelling, subsequent appointments and illustrative geographic movement; produce charts and Power BI exports (05_model.py–11_star_schema.py).
7. Reconcile dashboard totals with outputs and document assumptions and exceptions.
Critical limitations
- No reliable supply-side denominator: the data does not establish daily contracted/available technician hours, annual leave, sickness, shift changes or van downtime. Planned hours ÷ technician counts is not utilisation or spare capacity.
- Home territory is a roster attribute, not necessarily the only legitimate operating territory. An outside-home assignment alone is not evidence of inefficiency.
- Actual Start records a timestamp; it is not proof of arrival, actual travel, a wasted visit or time recovered. Planned appointment hours attached to failures are exposure, not financial or labour savings.
- Effective appointment dates can fall back from Appointment Date to Actual Start, then SchedStartTime; DateSource preserves which source was used.
- The extracted data has missing and specially coded territories, missing lead times and 13 operational appointments without a final recorded status. Their handling differs by analysis; modelling uses resolved outcomes with mapped regions.
- Geographic movement estimates use straight-line distances between observable started appointments and an illustrative 30 km/h assumption. They exclude depot/home legs and are not actual driven distances, travel hours or utilisation; do not use as a core dashboard KPI.
- A later appointment for the same vehicle can be unrelated to the earlier failure. The Created-Date-bounded extract also limits observed follow-up.
- Descriptive associations (e.g., failure rate by booking lead time) are not causal estimates.
Project structure
src/
  00_convert.py           Raw files -> parquet (run once for new source data)
  01_clean.py             Cleaning, date/source flags and territory joins
  02_demand.py            Demand and workload ranking
  03_capacity.py          Observed resource activity and coverage diagnostics
  04_opportunity_loss.py  Failures, cancellations, lead-time and exposure
  05_model.py             Time-split logistic-regression baseline
  06_recommendation.py    Stock-failure evidence and pilot proposal
  07_travel.py            Illustrative geographic-movement analysis
  08_first_time_fix.py    Observed later same-vehicle appointment proxy
  09_forecast.py          Exploratory weekday forecast baseline
  10_charts.py            Supporting charts
  11_star_schema.py       Power BI CSV export

data/processed/            Converted and cleaned parquet (may contain sensitive data)
outputs/tables/            Validation and analytical CSVs
outputs/charts/            Four supplementary PNG charts
outputs/powerbi/           Power BI-ready CSVs, plus any archived PBIX
KwikFit_Theme.json        Blue/yellow dashboard theme
Key reproducibility outputs include cleaning_log.csv, territory_effective.csv, booking_lead_time_audit.csv, regional_failure_exposure.csv, recommendation_evidence.csv, model_results.csv and first_time_fix.csv.
How to run (Windows Command Prompt)
From the repository root:
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy pyxlsb openpyxl pyarrow matplotlib scikit-learn
To regenerate from the original three source extracts, place them in the locations/formats expected by src/00_convert.py, then run the following in order. Conversion is unnecessary if the supplied data/processed/*_raw.parquet files already exist.
python src\00_convert.py
python src\01_clean.py
python src\02_demand.py
python src\03_capacity.py
python src\04_opportunity_loss.py
python src\05_model.py
python src\08_first_time_fix.py
python src\06_recommendation.py
python src\07_travel.py
python src\09_forecast.py
python src\10_charts.py
python src\11_star_schema.py
08_first_time_fix.py must run before 11_star_schema.py so the repeat eligibility flags are exported correctly. 07_travel.py must run before 11_star_schema.py so the revised technician-day table is available. This sequence regenerates files, so keep a copy of validated outputs before rerunning on changed input data.
Power BI
The Power BI model uses outputs/powerbi/fact_appointments.csv, dim_technician.csv, dim_territory.csv and fact_technician_day.csv, plus a date dimension. The report comprises Demand & Technician Deployment and Service Disruption & Planning Actions pages. Local Power Query CSV paths may need updating on another computer. Refresh and verify 47,747 appointment rows and 6,649 technician-day rows before interpreting visuals.
The final dashboard file is the separately saved, user-validated Power BI report (if provided with the assessment). Do not assume a PBIX embedded in a previously created code ZIP is that final version.
Confidentiality
These are employer-provided operational sample data. Treat raw data, technician details, customer information and vehicle identifiers as confidential. Avoid publishing raw extracts or sensitive records in a public Git repository. The Power BI export omits raw vehicle-registration identifiers; the raw/processed files may still contain them.
Scope: An evidence-led assessment and practical planning proposal, not a production utilisation system, a reliable travel measurement system or a deployed prediction service.