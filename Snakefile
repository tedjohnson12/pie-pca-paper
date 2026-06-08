rule table_all:
    output:
        "src/tex/output/tab.txt"
    script:
        "src/scripts/write_table.py"
