rule toi519_radius:
    output:
        "src/tex/figures/toi519_retrieval_red_chi_square_radius.pdf"
    script:
        "src/scripts/toi519_radius2d.py"
rule toi519_radius_eclipse_null:
    output:
        "src/tex/figures/toi519_retrieval_red_chi_square_radius_eclipse_null.pdf"
    script:
        "src/scripts/toi519_radius2d_eclipse_null.py"
rule toi519_radius_eclipse_half:
    output:
        "src/tex/figures/toi519_retrieval_red_chi_square_radius_eclipse_half.pdf"
    script:
        "src/scripts/toi519_radius2d_eclipse_half.py"
rule toi_table:
    output:
        "src/tex/output/toi519.txt"
    script:
        "src/scripts/toi519_run.py"