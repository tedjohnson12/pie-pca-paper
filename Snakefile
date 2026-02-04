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
rule toi519_radius_half:
    output:
        "src/tex/figures/toi519_retrieval_red_chi_square_radius_half.pdf"
    script:
        "src/scripts/toi519_radius2d_half.py"
rule toi_table:
    output:
        "src/tex/output/toi519.txt"
    script:
        "src/scripts/toi519_run.py"
rule pcb_radius_null:
    output:
        "src/tex/figures/proxb_retrieval_red_chi_square_radius_null.pdf"
    script:
        "src/scripts/proxb_radius2d_null.py"
rule pcb_radius_half:
    output:
        "src/tex/figures/proxb_retrieval_red_chi_square_radius_half.pdf"
    script:
        "src/scripts/proxb_radius2d_half.py"
rule pcb_radius_full:
    output:
        "src/tex/figures/proxb_retrieval_red_chi_square_radius_full.pdf"
    script:
        "src/scripts/proxb_radius2d_full.py"
rule gj_table:
    output:
        "src/tex/output/gj876.txt"
    script:
        "src/scripts/gj876_run.py"
rule pcb_table:
    output:
        "src/tex/output/proxb.txt"
    script:
        "src/scripts/proxb_run.py"
