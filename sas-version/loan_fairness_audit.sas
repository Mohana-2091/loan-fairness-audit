/* ============================================================
   Loan Fairness Audit — SAS version (corrected)
   ============================================================ */

data loan_data;
    call streaminit(42);
    do id = 1 to 5000;

        array race_opts[3] $10 _temporary_ ('groupA' 'groupB' 'groupC');
        protected_race = race_opts[ceil(rand('UNIFORM') * 3)];

        array sex_opts[2] $6 _temporary_ ('male' 'female');
        protected_sex = sex_opts[ceil(rand('UNIFORM') * 2)];

        income          = round(exp(rand('NORMAL', 11, 0.5)));
        loan_amount     = round(exp(rand('NORMAL', 12, 0.4)));
        credit_score    = round(rand('NORMAL', 680, 60));
        debt_to_income  = round(rand('UNIFORM') * 50, 0.1);
        loan_to_value   = round(rand('UNIFORM') * 40 + 60, 0.1);

        proxy_shift = ifc(protected_race = 'groupA', 15,
                      ifc(protected_race = 'groupB', -5, -10));
        neighborhood_income_tier = round(rand('NORMAL', 50 + proxy_shift, 12));

        risk_score = 0.01*credit_score - 0.05*debt_to_income
                     - 0.03*loan_to_value + 0.00002*income
                     + 0.02*neighborhood_income_tier
                     + rand('NORMAL', 0, 8);

        output;
    end;
    drop proxy_shift;
run;

proc means data=loan_data noprint;
    var risk_score;
    output out=med_stat median=median_risk;
run;
data _null_;
    set med_stat;
    call symputx('med_risk', median_risk);
run;

data loan_data;
    set loan_data;
    length approved $3;
    if risk_score > &med_risk then approved = 'yes';
    else approved = 'no';
run;

proc surveyselect data=loan_data out=split_data
     samprate=0.7 outall seed=42;
run;

data train test;
    set split_data;
    if selected = 1 then output train;
    else output test;
run;

title "Loan Approval Model (Protected Attributes Excluded)";
proc logistic data=train outmodel=loan_model;
    model approved(event='yes') = income loan_amount credit_score
          debt_to_income loan_to_value neighborhood_income_tier;
    score data=test out=scored_test;
    roc; roccontrast;
run;

/* --- FIX: sort both datasets by ID before merging --- */
proc sort data=scored_test;
    by id;
run;

proc sort data=test;
    by id;
run;

data audit_data;
    merge scored_test test(keep=id protected_race protected_sex);
    by id;
    predicted_approved = (P_yes >= 0.5);
run;

title "Approval Rate by Protected Group (Race) — Four-Fifths Rule Check";
proc sql;
    create table race_approval_rates as
    select protected_race,
           mean(predicted_approved) as approval_rate format=percent8.1
    from audit_data
    group by protected_race;

    select *,
           approval_rate / (select max(approval_rate) from race_approval_rates)
               as four_fifths_ratio format=percent8.1,
           case when calculated four_fifths_ratio < 0.80
                then 'FAILS four-fifths rule'
                else 'Passes'
                end as fairness_flag
    from race_approval_rates;
quit;

proc sql;
    create table group_thresholds as
    select protected_race,
           median(P_yes) as base_threshold
    from audit_data
    group by protected_race;
quit;

/* --- FIX: sort both datasets by protected_race before merging --- */
proc sort data=audit_data;
    by protected_race;
run;

proc sort data=group_thresholds;
    by protected_race;
run;

data corrected_audit;
    merge audit_data group_thresholds;
    by protected_race;
    corrected_approved = (P_yes >= base_threshold - 0.05);
run;

title "Approval Rate by Race — AFTER Equalized-Odds-Style Correction";
proc sql;
    create table corrected_rates as
    select protected_race,
           mean(corrected_approved) as approval_rate_corrected format=percent8.1
    from corrected_audit
    group by protected_race;

    select *,
           approval_rate_corrected /
               (select max(approval_rate_corrected) from corrected_rates)
               as four_fifths_ratio_corrected format=percent8.1
    from corrected_rates;
quit;

/* --- FIX: single summary row instead of repeated-row bug --- */
title "Recall Before vs. After Correction (Approved='yes' Actuals)";
proc sql;
    select
        mean(predicted_approved) as recall_before format=percent8.1,
        mean(corrected_approved) as recall_after format=percent8.1
    from corrected_audit
    where approved='yes';
quit;

title;
