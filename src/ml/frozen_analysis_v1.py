"""Pre-outcome confirmatory contrasts for the bounded point-regression study."""

BOOTSTRAP = {"resamples":10000,"seed":720901,"method":"crossed_paired_percentile",
             "quantile_method":"linear","primary_alpha":.025,"secondary_family_alpha":.025}


def contrast_specification():
    pairs=[("primary","R","G",.025),
        ("secondary","R","C",.005),("secondary","R","N",.005),("secondary","R","T",.005),
        ("secondary","R+MMD","G+MMD",.005),
        ("secondary","R+MMD-representative","G+MMD-representative",.005)]
    return [{"family":family,"candidate":candidate,"comparator":comparator,"alpha_one_sided":alpha,
        "meaningful_superiority_margin_arcsec":-.005,"canonical_ni_margin_arcsec":.005,
        "tail_ni_margin_arcsec":.010,"guardrails_required":True}
        for family,candidate,comparator,alpha in pairs]
