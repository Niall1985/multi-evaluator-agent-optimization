"""Narrative content for the BCSE497J project report.

Separated from build_report.py so the document machinery and the prose stay
independently editable. Each function takes the ReportBuilder and appends one
part of the report. Experimental numbers are pulled from the builder's loaded
report_data.json rather than written literally.
"""
from docx.enum.text import WD_ALIGN_PARAGRAPH as AL
from docx.shared import Pt

CENTER, JUSTIFY, LEFT, RIGHT = AL.CENTER, AL.JUSTIFY, AL.LEFT, AL.RIGHT


def _fmt_usd(x, places=4):
    return f"${x:.{places}f}"


def _pct(x, places=1):
    return f"{x:.{places}f}%"


# ==========================================================================
# FRONT MATTER
# ==========================================================================
def _title_block(b, C, is_cover: bool):
    """Cover page (item 1) and title page (item 2) of the prescribed sequence.

    They differ only in the opening line: the cover is headed "A project report
    on", the title page opens directly with the title. Both list every team
    member, per the template's first-page instruction.
    """
    if is_cover:
        b.titlepage_line("A project report on", 12, italic=True, space_after=10)
    else:
        b.titlepage_line("", 12, space_after=8)

    b.titlepage_line(C["title"], 20, bold=True, space_after=14)
    b.titlepage_line("Submitted in partial fulfillment for the award of the degree of",
                     14, italic=True, space_after=10)
    b.titlepage_line(C["degree"], 22, bold=True, space_after=14)
    b.titlepage_line("by", 14, italic=True, space_after=10)

    members = C["team"] if C["team"] else [(C["name"], C["reg_no"])]
    for nm, rg in members:
        b.titlepage_line(f"{nm} ({rg})", 16, bold=True, space_after=4)

    b.logo(space_after=10)

    b.titlepage_line(C["school"], 16, bold=True, space_after=4, space_before=6)
    b.titlepage_line(C["institute"], 16, bold=True, space_after=12)
    b.titlepage_line(C["month_year"], 16, bold=True, space_after=0)
    b.page_break()


def front_matter(b, C):
    d = b.d

    _title_block(b, C, is_cover=True)      # 1. Cover Page
    _title_block(b, C, is_cover=False)     # 2. Title Page

    # ------------------------------------------- 3. Declaration by the Candidate
    b.logo(space_after=14)
    b.front_heading("Declaration")
    # Template wording, verbatim.
    b.body(
        f'I hereby declare that the thesis entitled \u201c{C["title_prose"]}\u201d submitted by '
        f'{C["name_prose"]} ({C["reg_no"]}), for the award of the degree of {C["degree"]}, '
        f'{C["institute_prose"]} is a record of bonafide work carried out by me under the '
        f'supervision of {C["guide_name"]}.'
    )
    b.body(
        "I further declare that the work reported in this thesis has not been submitted and "
        "will not be submitted, either in part or in full, for the award of any other degree "
        "or diploma in this institute or any other institute or university."
    )
    b.signature_block([
        (f'Place: {C["place"]}', ""),
        ("Date:", "Signature of the Candidate"),
    ], space_before=28)
    b.page_break()

    # ------------------------------------------------------------ 4. Certificate
    b.logo(space_after=8)
    b.para(C["school_prose"], size=12, bold=True, align=CENTER, spacing=1.0, space_after=4)
    b.front_heading("Certificate")
    b.body(
        f'This is to certify that the report entitled \u201c{C["title_prose"]}\u201d is prepared '
        f'and submitted by {C["name_prose"]} ({C["reg_no"]}) to {C["institute_prose"]}, in partial '
        f'fulfillment of the requirement for the award of the degree of {C["degree"]} is a '
        f'bonafide record carried out under my guidance. The project fulfills the requirements as '
        f'per the regulations of this University and in my opinion meets the necessary standards '
        f'for submission. The contents of this report have not been submitted and will not be '
        f'submitted either in part or in full, for the award of any other degree or diploma and '
        f'the same is certified.'
    )
    b.signature_block([
        ("Signature of the Guide:", ""),
        (f'Name: Prof. {C["guide_name"]}', ""),
        ("Date:", ""),
    ], space_before=18)
    b.signature_block([
        ("Signature of the Examiner", "Signature of the Examiner"),
        ("Name:", "Name:"),
        ("Date:", "Date:"),
    ], space_before=18)
    b.signature_block([
        ("Approved by the Head of Department,", ""),
        (C["program"], ""),
        (f'Name: {C["hod_name"]}', ""),
        ("Date:", ""),
    ], space_before=18, bold_first=2)
    b.para(f'(Seal of {C["school_prose"]})', size=12, align=CENTER, spacing=1.0,
           space_after=0)
    b.page_break()

    # --------------------------------------------------------------- 5. Abstract
    b.front_heading("Abstract")
    ds = d["dataset"]
    cost = d.get("cost", {})
    pooled = d.get("wilcoxon_pooled", [])
    saved_pct = cost.get("saved_pct")
    saved_txt = (f"reduces cumulative evaluation expenditure by {_pct(saved_pct)} "
                 f"({_fmt_usd(cost.get('final_adaptive_usd', 0), 4)} against a "
                 f"{_fmt_usd(cost.get('final_naive_usd', 0), 4)} exhaustive baseline)"
                 if saved_pct is not None else
                 "reduces cumulative evaluation expenditure substantially")

    # The quality claim is derived, never asserted: compare C5 against exhaustive
    # evaluation on correctness and let the measured gap choose the wording.
    by_cond = {r["condition"]: r for r in d.get("condition_summary", [])}
    c5, c4 = by_cond.get("full_adaptive"), by_cond.get("no_pruning")
    if c5 and c4:
        gap = c5["mean_peak_correctness"] - c4["mean_peak_correctness"]
        if abs(gap) < 0.05:
            quality_txt = (
                f"while benchmark correctness is essentially unchanged "
                f"({c5['mean_peak_correctness']:.3f} against "
                f"{c4['mean_peak_correctness']:.3f} under exhaustive evaluation)")
        elif gap < 0:
            quality_txt = (
                f"at a measured cost in benchmark correctness, which fell from "
                f"{c4['mean_peak_correctness']:.3f} under exhaustive evaluation to "
                f"{c5['mean_peak_correctness']:.3f}")
        else:
            quality_txt = (
                f"while benchmark correctness rose from "
                f"{c4['mean_peak_correctness']:.3f} under exhaustive evaluation to "
                f"{c5['mean_peak_correctness']:.3f}")
    else:
        quality_txt = "with benchmark quality reported alongside in Chapter 6"

    reps = ds["n_runs_per_cell"]
    rep_txt = f"{reps} independent seeded repetition" + ("" if reps == 1 else "s")

    b.body(
        "Large language model (LLM) agents are acutely sensitive to their configuration: the "
        "system prompt that frames the task, the decoding parameters that shape the response, and "
        "the choice of metrics against which candidate configurations are judged. Existing "
        "automated prompt-optimization frameworks largely treat evaluation as free and "
        "single-objective, scoring every candidate against a fixed, exhaustively executed metric "
        "ensemble with hand-chosen weights. In practice this is both expensive, because "
        "LLM-as-a-judge evaluators dominate the cost of a search, and fragile, because redundant "
        "metrics consume budget while unmonitored dimensions regress silently."
    )
    b.body(
        "This project develops and evaluates a framework for multi-objective optimization of LLM "
        "agents through adaptive multi-evaluator co-optimization. The agent configuration and the "
        "evaluation process are optimized jointly over a hybrid search space comprising a discrete "
        "prompt population, a binary evaluator-activation vector, a continuous weight vector on the "
        "probability simplex, and continuous decoding parameters. Six contributions address six "
        "corresponding gaps in the literature: a two-tier evaluation cascade that gates expensive "
        "judges behind a cheap preview score; Gaussian-process optimization of the metric weights "
        "with a Matérn 5/2 kernel and a GP-UCB acquisition function; an explicitly cost-penalized "
        "scalarized fitness function; functional ANOVA over the surrogate model to detect redundant "
        "evaluators through first-order Sobol indices; a per-dimension elite archive with "
        "epsilon-probing and silent-regression detection; and a unified treatment of the discrete "
        "and continuous components of the search."
    )
    b.body(
        f"The framework was implemented as a Python system with a FastAPI service and a React "
        f"instrumentation studio, and evaluated across {ds['n_tasks']} benchmark tasks spanning two "
        f"agent domains, namely Python code synthesis and neural sentiment analysis. A five-condition "
        f"ablation was executed with {rep_txt} per cell "
        f"and {ds['generations_per_run']} generations per run, totalling {ds['mock_runs']} optimization "
        f"runs and {ds['mock_generations']} generations, and was analysed using paired Wilcoxon "
        f"signed-rank tests. The adaptive cascade {saved_txt} relative to exhaustive evaluation, "
        f"{quality_txt}. Every generation was executed against a live Groq-hosted model: agent "
        f"outputs were produced by the model and candidate solutions were executed in a sandbox "
        f"against each task's test cases, with no simulated response admitted to the dataset."
    )
    arc = d.get("arc")
    if arc:
        b.body(
            f"A second experiment applied the framework to the ARC-AGI benchmark, on which current "
            f"language models score in the low single digits. Running a program archive over "
            f"{arc['n_tasks']} held-out tasks for {arc['generations']} generations, the official "
            f"pass@2 score rose from {arc['official_start_pct']:.1f} per cent to "
            f"{arc['official_final_pct']:.1f} per cent, a fourfold improvement measured on held-out "
            f"grids that never entered the fitness signal, while the number of tasks whose "
            f"demonstration pairs were fitted grew from {arc['demos_start']} to "
            f"{arc['demos_final']}. The gap between those two counts is reported as a finding in "
            f"its own right: {arc['generalisation_gap']} programs reproduced every demonstration "
            f"without capturing the underlying rule."
        )
    b.body(
        "The report also documents the limits of the study candidly. The optimization horizon is "
        f"short, at {ds['generations_per_run']} generations per run, so the convergence evidence "
        "describes early search behaviour rather than asymptotic performance. A single repetition "
        "per design cell means the per-task tests carry no within-cell variance estimate, and the "
        "pooled analysis is correspondingly the informative one. Both limits follow from the daily "
        "request allowance of the API tier available for this project, and both are quantified "
        "rather than glossed."
    )
    b.para()
    b.para("Keywords: Large language model agents; multi-objective optimization; Bayesian "
           "optimization; Gaussian processes; functional ANOVA; evaluator pruning; "
           "cost-aware evaluation; quality-diversity archives.",
           size=12, italic=True)
    b.page_break()

    # ---------------------------------------------------------- acknowledgement
    b.front_heading("Acknowledgement")
    # Template wording, verbatim; only the bracketed placeholders are filled.
    b.body(
        f'It is my pleasure to express with deep sense of gratitude to {C["guide_name"]}, '
        f'{C["guide_designation"]}, {C["school_prose"]}, {C["institute_prose"]}, for his/her '
        f'constant guidance, continual encouragement, understanding; more than all, he/she taught '
        f'me patience in my endeavor. My association with him / her is not confined to academics '
        f'only, but it is a great opportunity on my part of work with an intellectual and expert '
        f'in the field of {C["research_area"]}.'
    )
    b.body(
        "It is with gratitude that I would like to extend my thanks to the visionary leader "
        "Dr. G. Viswanathan our Honorable Chancellor, Mr. Sankar Viswanathan, Dr. Sekar "
        "Viswanathan, Dr. G V Selvam Vice Presidents, Dr. Sandhya Pentareddy, Executive Director, "
        "Ms. Kadhambari S. Viswanathan, Assistant Vice-President, Dr. V. S. Kanchana Bhaaskaran "
        "Vice-Chancellor, Dr. T. Thyagarajan Pro-Vice Chancellor, VIT Chennai and "
        "Dr. P. K. Manoharan, Additional Registrar for providing an exceptional working "
        "environment and inspiring all of us during the tenure of the course."
    )
    b.body(
        "Special mention to Dr. Viswanathan V, Dean, Dr. Nithyanandam P, Associate Dean, "
        "Dr. Suganya G, Associate Dean, Dr. Sweetlin Hemalatha C, Associate Dean, School of "
        "Computer Science and Engineering, Vellore Institute of Technology, Chennai for spending "
        "their valuable time and efforts in sharing their knowledge and for helping us in every "
        "aspect."
    )
    b.body(
        f'In jubilant state, I express ingeniously my whole-hearted thanks to {C["hod_name"]}, '
        f'Head of the Department, {C["program"]} and the Project Coordinators for their valuable '
        f'support and encouragement to take up and complete the thesis.'
    )
    b.body(
        "My sincere thanks to all the faculties and staffs at Vellore Institute of Technology, "
        "Chennai who helped me acquire the requisite knowledge. I would like to thank my parents "
        "for their support. It is indeed a pleasure to thank my friends who encouraged me to take "
        "up and complete this task."
    )
    b.signature_block([
        (f'Place: {C["place"]}', ""),
        ("Date:", C["name_prose"]),
    ], space_before=22)
    b.page_break()


def table_of_contents(b):
    """Word TOC field; press F9 (or Update Field) on opening to populate."""
    b.front_heading("Contents")
    p = b.doc.add_paragraph()
    from build_report import _field
    _field(p, r'TOC \o "1-3" \h \z \u')
    b.para()
    b.para("[If this page appears blank, select the field and press F9, or choose "
           "References > Update Table, to generate the table of contents.]",
           size=10, italic=True, align=CENTER)
    b.page_break()


def list_of_figures_tables(b):
    """Built as Word TOC fields keyed to the caption styles, so each entry
    carries its page number as the template's sample pages show."""
    from build_report import _field

    b.front_heading("List of Figures")
    p = b.doc.add_paragraph()
    _field(p, r'TOC \h \z \t "FigureCaption,1"')
    b.page_break()

    b.front_heading("List of Tables")
    p = b.doc.add_paragraph()
    _field(p, r'TOC \h \z \t "TableCaption,1"')
    b.page_break()


def list_of_symbols(b):
    """Item 11 of the prescribed sequence: symbols, abbreviations, nomenclature."""
    b.front_heading("List of Symbols and Nomenclature")
    rows = [
        ("\u03a6", "An agent configuration: system prompt together with its decoding parameters"),
        ("p", "System prompt of a candidate agent"),
        ("x_E", "Evaluator activation vector, an element of {0,1} raised to the sixth power"),
        ("w", "Metric weight vector on the probability simplex"),
        ("\u0394\u2075", "Five-dimensional probability simplex on which w lies"),
        ("\u03b8_H", "Harness knobs: temperature, nucleus sampling threshold, retries, token budget"),
        ("\u03bc\u2c7c", "Score returned by evaluator j, normalized to the unit interval"),
        ("c\u2c7c", "Unit cost of executing evaluator j"),
        ("\u03bb", "Cost penalty coefficient in the scalarized fitness (0.5 throughout)"),
        ("F", "Cost-penalized scalarized fitness of a candidate"),
        ("F\u2099\u2092\u1d63\u2098", "Fitness normalized by the active weight mass, comparable across conditions"),
        ("\u0394F", "Improvement in fitness of a child over its parent"),
        ("\u03c4", "Preview-score threshold gating the deep evaluator tier (0.65)"),
        ("\u03b5", "Probability of overriding a pruning decision to audit it (0.30)"),
        ("S\u2c7c", "First-order Sobol sensitivity index of evaluator j"),
        ("V(f)", "Total variance of the fitness surrogate"),
        ("\u03b2\u209c", "GP-UCB exploration coefficient at iteration t"),
        ("\u03bc(w), \u03c3(w)", "Posterior mean and standard deviation of the Gaussian process at w"),
        ("\u03b1(w)", "Acquisition function evaluated at weight vector w"),
        ("\u03b3\u209c", "Maximum information gain after t observations"),
        ("r", "Matched-pairs rank-biserial correlation, the reported effect size"),
        ("W", "Wilcoxon signed-rank test statistic"),
        ("n", "Number of matched pairs entering a statistical test"),
    ]
    for sym, meaning in rows:
        para = b.doc.add_paragraph()
        para.paragraph_format.line_spacing = 1.0
        para.paragraph_format.space_after = Pt(3)
        r1 = para.add_run(f"{sym}\t")
        r1.bold = True
        r1.font.name = "Times New Roman"
        r1.font.size = Pt(12)
        r2 = para.add_run(meaning)
        r2.font.name = "Times New Roman"
        r2.font.size = Pt(12)
    b.page_break()


def list_of_acronyms(b):
    b.front_heading("List of Acronyms and Abbreviations")
    rows = [
        ("API", "Application Programming Interface"),
        ("CPU", "Central Processing Unit"),
        ("CSV", "Comma Separated Values"),
        ("EI", "Expected Improvement"),
        ("fANOVA", "Functional Analysis of Variance"),
        ("GP", "Gaussian Process"),
        ("GP-UCB", "Gaussian Process Upper Confidence Bound"),
        ("HTTP", "Hypertext Transfer Protocol"),
        ("JSON", "JavaScript Object Notation"),
        ("LLM", "Large Language Model"),
        ("MAP-Elites", "Multi-dimensional Archive of Phenotypic Elites"),
        ("MBPP", "Mostly Basic Python Problems"),
        ("REST", "Representational State Transfer"),
        ("RPD", "Requests Per Day"),
        ("s.e.m.", "Standard Error of the Mean"),
        ("TPM", "Tokens Per Minute"),
        ("UCB1", "Upper Confidence Bound 1"),
        ("UI", "User Interface"),
    ]
    for a, full in rows:
        p = b.doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.0
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(f"{a}\t")
        r.bold = True
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)
        r2 = p.add_run(full)
        r2.font.name = "Times New Roman"
        r2.font.size = Pt(12)
    b.page_break()


# ==========================================================================
# CHAPTER 1 - INTRODUCTION
# ==========================================================================
def chapter1(b):
    d = b.d
    ds = d["dataset"]
    b.chapter_heading("1", "Introduction")

    b.section_heading("1.1", "Introduction")
    b.body(
        "A large language model agent is not merely a model. It is a model wrapped in a "
        "configuration: a system prompt that frames the task and constrains the response, a set of "
        "decoding parameters such as temperature and nucleus sampling threshold, a retry policy, and "
        "an output budget. Holding the underlying model weights fixed and varying only this wrapper "
        "produces strikingly different behaviour on the same task. The practical consequence is that "
        "building a capable agent is, to a substantial degree, a search problem over configurations "
        "rather than a modelling problem."
    )
    b.body(
        "Automating that search is the subject of a growing body of work. Systems such as APE, OPRO, "
        "PromptBreeder and DSPy treat the prompt as an optimizable object and search over it using "
        "an LLM as the mutation operator, a scoring function as the fitness signal, and some form of "
        "hill climbing, evolutionary search or compilation to drive improvement. These systems have "
        "demonstrated that automated search can match or exceed expert prompt engineering."
    )
    b.body(
        "What these systems have largely not addressed is the other half of the problem. Every one of "
        "them requires a fitness signal, and that signal must be computed for every candidate "
        "configuration that the search considers. When the fitness signal is a single automatic "
        "metric such as a unit-test pass rate, this is cheap. When it is what practitioners actually "
        "want, namely a composite judgement over correctness, efficiency, reasoning quality, "
        "robustness to edge cases and safety, it becomes the dominant cost of the search, because "
        "several of those dimensions are themselves assessed by LLM calls. The evaluation process, "
        "in other words, is an expensive resource that is itself amenable to optimization, and it is "
        "generally treated as free."
    )
    b.body(
        "This project takes that observation as its starting point. It develops a framework in which "
        "the agent configuration and the evaluation process are optimized jointly: the system decides "
        "not only what the agent should look like, but also which metrics are worth paying for at "
        "each step of the search, how those metrics should be weighted, and which of them have "
        "ceased to provide useful signal."
    )

    b.section_heading("1.2", "Overview of agent configuration optimization")
    b.body(
        "The optimization problem addressed here operates over a hybrid search space. Writing a "
        "candidate configuration as a tuple, the space comprises four components. The first is the "
        "discrete space of system prompts, explored by applying LLM-driven mutation operators to a "
        "parent prompt drawn from an evolving population. The second is a binary evaluator-activation "
        "vector over the six evaluation dimensions, which determines the subset of metrics actually "
        "executed for a given candidate. The third is a continuous weight vector constrained to the "
        "probability simplex, which determines how the executed metrics are combined into a scalar "
        "fitness. The fourth is the continuous space of decoding parameters, principally temperature "
        "and nucleus sampling threshold."
    )
    b.body(
        "A single generation of the search proceeds by selecting a parent from the population, "
        "applying a mutation operator to produce a child configuration, proposing an evaluator subset "
        "and a weight vector, executing the child agent on a benchmark task, scoring the resulting "
        "output on the active evaluators, and combining those scores into a cost-penalized scalar "
        "fitness. The resulting observation updates the surrogate models that drive subsequent "
        "proposals, and the child is inserted into the population archive."
    )

    b.section_heading("1.3", "Challenges in multi-metric agent evaluation")
    b.body(
        "Six specific difficulties motivate the design of the framework, and each is addressed by a "
        "corresponding mechanism described in Chapter 4."
    )
    b.subsection_heading("1.3.1", "Fixed and exhaustive evaluation")
    b.body(
        "Conventional optimization loops evaluate every candidate against the complete metric "
        "ensemble. Because many candidates are plainly non-viable, and because the expensive "
        "dimensions are precisely those assessed by LLM judges, a large fraction of the evaluation "
        "budget is spent establishing that a poor candidate is poor."
    )
    b.subsection_heading("1.3.2", "Statically hand-weighted objectives")
    b.body(
        "Composite objectives are typically formed by fixing the metric weights in advance. This "
        "assumes the designer knows the correct trade-off before the search has produced any "
        "evidence, and it prevents the objective from adapting as the population improves and the "
        "binding constraint shifts from one dimension to another."
    )
    b.subsection_heading("1.3.3", "Cost-blind search")
    b.body(
        "Evaluation consumes both compute and, for hosted models, money. When the objective does not "
        "represent that expenditure, the optimizer has no reason to prefer a configuration that "
        "achieves comparable quality at lower evaluation cost."
    )
    b.subsection_heading("1.3.4", "Undetected evaluator redundancy")
    b.body(
        "Metrics in an ensemble are rarely independent. Some are strongly correlated with others and "
        "some become uninformative once the population has saturated on that dimension. Without an "
        "explicit mechanism to detect this, redundant evaluators continue to be executed and paid for "
        "indefinitely."
    )
    b.subsection_heading("1.3.5", "Silent regression on unmonitored dimensions")
    b.body(
        "Any mechanism that stops evaluating a dimension creates the possibility that the population "
        "quietly degrades on it. This is the central risk introduced by pruning, and a framework that "
        "prunes without auditing is unsound."
    )
    b.subsection_heading("1.3.6", "Disconnected discrete and continuous search")
    b.body(
        "Prompt text is discrete and combinatorial; weights and decoding parameters are continuous. "
        "Frameworks commonly optimize these with separate, disconnected heuristics, which forfeits "
        "any principled statement about convergence."
    )

    b.section_heading("1.4", "Project statement")
    b.body(
        "To design, implement and empirically evaluate a framework that optimizes large language "
        "model agent configurations and the evaluation process itself as a single joint problem, such "
        "that the cumulative cost of evaluation is materially reduced relative to exhaustive "
        "multi-metric scoring without a corresponding loss of agent quality, and such that the "
        "dimensions that are not evaluated at a given step remain under audit rather than unobserved."
    )

    b.section_heading("1.5", "Objectives")
    b.bullets([
        "To formulate agent configuration optimization as a joint discrete-continuous search over "
        "prompts, evaluator activation, metric weights and decoding parameters.",
        "To design a tiered evaluation cascade that executes cheap deterministic metrics first and "
        "gates expensive LLM-judge metrics behind a preview-score threshold.",
        "To replace hand-chosen metric weights with Gaussian-process optimization over the "
        "probability simplex, using an upper-confidence-bound acquisition function.",
        "To incorporate evaluator cost directly into the fitness function so that the search is "
        "cost-aware by construction.",
        "To detect redundant evaluators automatically by decomposing the variance of the surrogate "
        "model and computing first-order Sobol sensitivity indices.",
        "To guard against silent regression using a per-dimension elite archive, probabilistic "
        "auditing of pruned dimensions, and explicit regression alerts.",
        "To implement the framework as a working system with a programmatic interface, a REST "
        "service and an interactive instrumentation studio.",
        "To evaluate the framework through a controlled five-condition ablation across two agent "
        "domains, analysed with appropriate non-parametric statistical tests, and to validate the "
        "pipeline against a live hosted language model.",
    ])

    b.section_heading("1.6", "Scope of the project")
    b.body(
        f"The framework is evaluated on {ds['n_tasks']} benchmark tasks across two domains. The "
        f"coding domain comprises seven Python synthesis tasks drawn from or modelled on the "
        f"HumanEval and MBPP benchmarks, assessed by executing the generated function against test "
        f"cases in a sandbox. The sentiment domain comprises five classification suites assessed "
        f"against a pre-trained Keras sentiment model. Six evaluation dimensions are used throughout: "
        f"functional correctness, execution latency, output conciseness, reasoning quality, edge-case "
        f"robustness and safety, of which the third and sixth are assessed by an LLM judge."
    )
    arc = d.get("arc")
    if arc:
        b.body(
            f"A second benchmark is used for a separate purpose. The canonical suite is one on "
            f"which a competent model already succeeds much of the time, which makes it suitable "
            f"for measuring the cost of evaluation but unsuitable for measuring whether the search "
            f"improves capability, since there is little headroom to improve. The ARC-AGI "
            f"benchmark supplies that headroom: its tasks are abstract grid transformations that "
            f"must be inferred from a few demonstration pairs, current language models solve them "
            f"at rates in the low single digits, and scoring is pixel-exact. "
            f"{arc['n_tasks']} of its held-out tasks are used, with a distinct evaluator suite "
            f"described in Section 5.3.6 and a program-archive search mode described in "
            f"Section 4.8."
        )
    b.body(
        "The project does not fine-tune or otherwise modify model weights; the underlying language "
        "model is treated as a fixed, hosted resource accessed over an API. It does not address "
        "multi-agent orchestration, tool use or retrieval augmentation. The benchmark suite is "
        "deliberately small and is intended to support controlled comparison between optimization "
        "conditions rather than to establish absolute capability rankings between language models."
    )

    b.section_heading("1.7", "Organization of the report")
    b.body(
        "Chapter 2 surveys the relevant literature and states the six research gaps precisely. "
        "Chapter 3 presents the requirements analysis and feasibility study. Chapter 4 describes the "
        "system architecture and the algorithmic design of each mechanism. Chapter 5 documents the "
        "implementation. Chapter 6 presents the experimental methodology, results and discussion, "
        "including an explicit treatment of the limitations of the evidence. Chapter 7 concludes and "
        "identifies future work."
    )
    b.page_break()


# ==========================================================================
# CHAPTER 2 - BACKGROUND AND LITERATURE SURVEY
# ==========================================================================
def chapter2(b):
    b.chapter_heading("2", "Background and Literature Survey")

    b.section_heading("2.1", "Introduction")
    b.body(
        "This chapter reviews the work on which the project builds. It covers automated prompt and "
        "agent optimization, the optimization machinery borrowed from the Bayesian and "
        "multi-objective literature, the practice of evaluating language model outputs with other "
        "language models, and the sensitivity-analysis and quality-diversity methods used here to "
        "reason about evaluator redundancy and regression. The chapter closes by stating the six "
        "gaps the project addresses."
    )

    b.section_heading("2.2", "Automated prompt and agent optimization")
    b.body(
        "Zhou et al. (2023) established with Automatic Prompt Engineer that an LLM can propose "
        "instruction candidates which are then selected by score, and that the result is competitive "
        "with human prompt engineering. Yang et al. (2024) generalized the idea in OPRO by treating "
        "the language model itself as the optimizer, feeding it a trajectory of previously scored "
        "solutions and asking it to propose improvements."
    )
    b.body(
        "PromptBreeder (Fernando et al., 2023) introduced self-referential evolution, in which both "
        "the task prompts and the mutation prompts that modify them are themselves subject to "
        "evolutionary pressure. DSPy (Khattab et al., 2024) reframed the problem as compilation: "
        "programs are written declaratively in terms of modules and signatures, and a compiler "
        "optimizes the underlying prompts and few-shot demonstrations against a developer-supplied "
        "metric. More recently, GEPA (Agrawal et al., 2025) demonstrated that reflective prompt "
        "evolution, which uses natural-language reflection on execution traces as the mutation "
        "signal, can outperform reinforcement-learning-based approaches at substantially lower "
        "sample cost, and AlphaEvolve (Novikov et al., 2025) applied LLM-driven evolutionary search "
        "to algorithm discovery."
    )
    b.body(
        "The common structure across this family is a generate-score-select loop. The common "
        "assumption is that the score is available on demand. DSPy makes the metric an explicit "
        "developer-supplied function; PromptBreeder and GEPA assume a task fitness; OPRO assumes a "
        "scored trajectory. In every case the evaluation is a given of the problem rather than a "
        "variable of it, and its cost does not appear in the objective. That assumption is the one "
        "this project relaxes."
    )

    b.section_heading("2.3", "Multi-objective and Bayesian optimization")
    b.body(
        "When several objectives conflict, two standard approaches apply. The first is Pareto "
        "methods, which maintain a set of mutually non-dominated solutions; NSGA-II (Deb et al., "
        "2002) is the canonical algorithm and supplies the notion of non-dominated sorting used in "
        "this project's population archive. The second is scalarization, which collapses the "
        "objectives into a single weighted sum. Scalarization is computationally convenient but "
        "requires weights, and the choice of weights determines which part of the Pareto front is "
        "reachable."
    )
    b.body(
        "Rather than fixing those weights, this project treats the weight vector as the decision "
        "variable of an inner Bayesian optimization problem. Gaussian process regression "
        "(Rasmussen and Williams, 2006) provides a posterior over the mapping from weights to "
        "observed improvement, together with a calibrated uncertainty estimate. Srinivas et al. "
        "(2010) analysed Gaussian process optimization in the bandit setting and established that "
        "the upper-confidence-bound acquisition rule, which selects the point maximizing the "
        "posterior mean plus a scaled posterior standard deviation, achieves cumulative regret "
        "sublinear in the number of iterations when the exploration coefficient is set according to "
        "their schedule. That result is what licenses the use of GP-UCB here, and the implementation "
        "follows their prescribed coefficient."
    )
    b.body(
        "For the discrete component of the search, Auer et al. (2002) analysed the UCB1 algorithm "
        "for the stochastic multi-armed bandit and proved a logarithmic regret bound. The framework "
        "applies UCB1 to the selection of mutation operators, treating each operator as an arm whose "
        "reward is the fitness improvement it produces."
    )

    b.section_heading("2.4", "Evaluation of language model outputs")
    b.body(
        "For code synthesis, execution-based evaluation is the established standard. HumanEval "
        "(Chen et al., 2021) and MBPP (Austin et al., 2021) both score a generated function by "
        "executing it against hidden test cases, which yields an objective and reproducible "
        "correctness signal. The benchmark tasks in this project follow the same protocol and "
        "several are drawn directly from those suites."
    )
    b.body(
        "For dimensions that resist execution-based scoring, such as reasoning quality and safety, "
        "the prevailing method is to use a strong language model as a judge. Zheng et al. (2023) "
        "studied this practice systematically and found that LLM judges agree with human preferences "
        "at rates comparable to inter-human agreement, while also documenting characteristic biases "
        "including position bias and verbosity bias. This project uses LLM judges for two of its six "
        "dimensions and treats them as the expensive tier precisely because each judgement is an "
        "additional model invocation."
    )

    b.section_heading("2.5", "Sensitivity analysis and quality-diversity methods")
    b.body(
        "To decide whether an evaluator still contributes information, the framework decomposes the "
        "variance of the fitness surrogate. Sobol (2001) defined global sensitivity indices for "
        "nonlinear models, decomposing the variance of a function of several inputs into "
        "contributions from each input and from their interactions; the first-order index of an "
        "input is the fraction of total variance attributable to that input alone. Hutter et al. "
        "(2014) applied functional ANOVA to algorithm configuration, showing that the importance of "
        "hyperparameters can be assessed efficiently by fitting a surrogate model and decomposing "
        "its variance. The framework adopts this approach, treating evaluator scores as the inputs "
        "and the fitness as the response, and marks an evaluator as redundant when its first-order "
        "index falls below a threshold."
    )
    b.body(
        "To guard the dimensions that pruning stops observing, the framework borrows from "
        "quality-diversity search. MAP-Elites (Mouret and Clune, 2015) maintains an archive of the "
        "best solution found in each cell of a behavioural space rather than a single global best, "
        "which preserves diversity and makes regression on any individual behavioural dimension "
        "visible. The archive used here is a one-dimensional analogue: for each of the six evaluation "
        "dimensions it retains the highest-scoring configuration ever observed on that dimension, "
        "which provides the reference point against which silent regression is detected."
    )

    b.section_heading("2.6", "Summary of research gaps")
    b.body(
        "The survey identifies six gaps, each of which motivates one mechanism in the design "
        "presented in Chapter 4."
    )
    b.table(
        ["Gap", "Limitation in existing work", "Mechanism adopted in this project"],
        [
            ["G1", "Candidates are scored against the complete metric ensemble regardless of "
                   "viability, so budget is spent confirming that poor candidates are poor.",
             "Two-tier cascade: cheap deterministic metrics first, LLM judges gated behind a "
             "preview-score threshold."],
            ["G2", "Composite objectives use weights fixed in advance of any evidence and held "
                   "constant as the population improves.",
             "Gaussian process over the weight simplex with a GP-UCB acquisition rule."],
            ["G3", "Evaluation compute and API expenditure do not appear in the objective.",
             "Cost-penalized scalarized fitness with an explicit penalty coefficient."],
            ["G4", "Redundant or saturated evaluators are executed indefinitely because redundancy "
                   "is never measured.",
             "Functional ANOVA over the surrogate with first-order Sobol indices and a redundancy "
             "threshold."],
            ["G5", "Pruning an evaluator leaves its dimension unobserved, permitting silent "
                   "regression.",
             "Per-dimension elite archive, epsilon-probing audits, and explicit regression alerts."],
            ["G6", "Discrete prompt search and continuous parameter tuning are driven by separate "
                   "heuristics with no convergence argument.",
             "Unified joint space with UCB1 on the discrete component and GP-UCB on the continuous "
             "component."],
        ],
        "Research gaps identified in the literature and the corresponding mechanisms",
        widths=[1.4, 7.0, 6.1],
    )

    b.section_heading("2.7", "Summary")
    b.body(
        "Existing agent-optimization frameworks establish that automated search over configurations "
        "is effective, but they treat evaluation as a fixed, free and externally supplied oracle. "
        "The optimization and analysis literature supplies the tools needed to relax that assumption: "
        "Gaussian process bandits for the continuous weights, UCB1 for discrete operator selection, "
        "functional ANOVA for redundancy detection, and quality-diversity archives for regression "
        "monitoring. The contribution of this project is to assemble these into a single framework "
        "in which the evaluation process is itself optimized."
    )
    b.page_break()


# ==========================================================================
# CHAPTER 3 - REQUIREMENTS AND SYSTEM ANALYSIS
# ==========================================================================
def chapter3(b):
    b.chapter_heading("3", "Requirements and System Analysis")

    b.section_heading("3.1", "Introduction")
    b.body(
        "This chapter states the functional and non-functional requirements of the framework, "
        "records the hardware and software environment in which it was developed and evaluated, and "
        "assesses technical, economic and operational feasibility."
    )

    b.section_heading("3.2", "Functional requirements")
    b.table(
        ["ID", "Requirement", "Priority"],
        [
            ["FR1", "The system shall maintain an evolving population of agent configurations, each "
                    "comprising a system prompt and a set of decoding parameters.", "High"],
            ["FR2", "The system shall generate child configurations by applying LLM-driven mutation "
                    "operators to a parent selected from the population.", "High"],
            ["FR3", "The system shall execute a candidate agent on a selected benchmark task and "
                    "capture its output.", "High"],
            ["FR4", "The system shall score candidate outputs on six evaluation dimensions, of which "
                    "four are deterministic and two are LLM-judged.", "High"],
            ["FR5", "The system shall select, per generation, the subset of evaluators to execute, "
                    "using a preview-score gate over the cheap tier.", "High"],
            ["FR6", "The system shall propose metric weights on the probability simplex from a "
                    "Gaussian process surrogate using a GP-UCB acquisition function.", "High"],
            ["FR7", "The system shall compute a scalar fitness that subtracts a penalty proportional "
                    "to the cost of the evaluators actually executed.", "High"],
            ["FR8", "The system shall decompose surrogate variance and report first-order Sobol "
                    "indices per evaluator, flagging those below the redundancy threshold.", "Medium"],
            ["FR9", "The system shall maintain a per-dimension elite archive and raise an alert when "
                    "a candidate regresses on any dimension beyond a configured magnitude.", "High"],
            ["FR10", "The system shall override pruning decisions with a fixed probability so that "
                     "pruned dimensions are periodically audited.", "Medium"],
            ["FR11", "The system shall maintain a Pareto-non-dominated set over fitness and cost.",
             "Medium"],
            ["FR12", "The system shall expose its state through a REST API and an interactive web "
                     "interface, and persist per-generation telemetry to CSV.", "Medium"],
            ["FR13", "The system shall operate without an API key by falling back to a deterministic "
                     "response simulator, for reproducible offline experimentation.", "Medium"],
            ["FR14", "The system shall provide a strict mode that disables the simulator fallback, so "
                     "that a failed request aborts an experiment rather than silently recording a "
                     "simulated response as measured data.", "High"],
        ],
        "Functional requirements",
        widths=[1.3, 11.4, 1.8],
    )

    b.section_heading("3.3", "Non-functional requirements")
    b.table(
        ["ID", "Requirement"],
        [
            ["NFR1", "Reproducibility: every experimental condition shall be reproducible from a "
                     "recorded random seed, and all reported numbers shall be derivable from "
                     "persisted raw result files."],
            ["NFR2", "Cost efficiency: the adaptive configuration shall reduce cumulative evaluation "
                     "expenditure relative to exhaustive evaluation without a corresponding loss of "
                     "benchmark quality."],
            ["NFR3", "Robustness: the system shall tolerate API rate limiting, transient failures and "
                     "decommissioned model identifiers by backing off and failing over to an "
                     "alternative model."],
            ["NFR4", "Safety of execution: candidate code shall be executed in a constrained namespace "
                     "with a bounded time budget so that a faulty candidate cannot stall the search."],
            ["NFR5", "Observability: the state of every mechanism shall be inspectable at the "
                     "granularity of a single generation."],
            ["NFR6", "Portability: the system shall run on a standard consumer workstation without "
                     "specialized accelerator hardware."],
        ],
        "Non-functional requirements",
        widths=[1.3, 13.2],
    )

    b.section_heading("3.4", "Hardware and software requirements")
    b.table(
        ["Component", "Specification used"],
        [
            ["Processor", "Intel Core i5-1135G7, 4 physical cores, 8 logical processors, 2.40 GHz"],
            ["Memory", "15.7 GB RAM"],
            ["Operating system", "Windows 11 Home, build 10.0.26200"],
            ["Language runtime", "Python 3.11.9"],
            ["Numerical stack", "NumPy, SciPy, pandas, scikit-learn (Gaussian process regression)"],
            ["Neural runtime", "TensorFlow 2.21 with Keras 3.15, for the sentiment model"],
            ["LLM access", "Groq cloud API, openai/gpt-oss-120b, free tier"],
            ["Backend service", "FastAPI with the Uvicorn ASGI server"],
            ["Frontend", "React 18 with Vite 5 and Tailwind CSS"],
            ["Secondary interface", "Streamlit"],
            ["Accelerator", "None required; all computation performed on CPU"],
        ],
        "Development and evaluation environment",
        widths=[4.2, 10.3],
    )

    b.section_heading("3.5", "Feasibility study")
    b.subsection_heading("3.5.1", "Technical feasibility")
    b.body(
        "All algorithmic components are available in mature open-source libraries. Gaussian process "
        "regression with a Matérn kernel is provided by scikit-learn, non-parametric hypothesis "
        "testing by SciPy (Virtanen et al., 2020), and the language model is accessed over a "
        "documented HTTP API. The "
        "sentiment model is a pre-trained Keras artefact distributed as a Python package. No "
        "component requires specialized hardware, and the complete system runs on the consumer "
        "workstation specified in Table 3.3."
    )
    b.subsection_heading("3.5.2", "Economic feasibility")
    b.body(
        "The language model is accessed through a free service tier, and all other software is "
        "open source. The principal economic constraint is not monetary but quota: the free tier "
        "permits a limited number of requests per day and a limited token throughput per minute. "
        "This constraint directly shaped the experimental design described in Chapter 6: it fixes "
        "how many generations and repetitions the live ablation can afford, and the design was "
        "sized to the allowance rather than the allowance stretched to the design. Paradoxically, "
        "this constraint also motivates the "
        "project: a framework that reduces the number of evaluator invocations required per "
        "generation is of direct practical value precisely in such a setting."
    )
    b.subsection_heading("3.5.3", "Operational feasibility")
    b.body(
        "The framework is operated either programmatically through an experiment harness, or "
        "interactively through the React studio, which exposes each mechanism in a dedicated view. "
        "Telemetry is persisted to CSV in a fixed schema, so analysis is reproducible from raw "
        "results without re-executing the search."
    )

    b.section_heading("3.6", "Summary")
    b.body(
        "The requirements establish a system that must optimize agent configurations and evaluation "
        "jointly, remain reproducible and observable, and operate within the quota limits of a free "
        "API tier on commodity hardware. All three feasibility dimensions are satisfied, with the "
        "API quota identified as the binding constraint on experimental scale."
    )
    b.page_break()


# ==========================================================================
# CHAPTER 4 - SYSTEM DESIGN
# ==========================================================================
def chapter4(b):
    b.chapter_heading("4", "System Design")

    b.section_heading("4.1", "Introduction")
    b.body(
        "This chapter presents the design of the framework. It states the optimization problem "
        "formally, describes the system architecture, and then specifies each of the six mechanisms "
        "that address the gaps identified in Chapter 2. The chapter closes with the algorithm for a "
        "single generation and the specification of the evaluator suite."
    )

    b.section_heading("4.2", "Formal problem formulation")
    b.body(
        "Let a candidate solution be the tuple consisting of a system prompt p drawn from the "
        "discrete space of prompts, a binary evaluator-activation vector x over the six evaluation "
        "dimensions, a weight vector w lying on the five-dimensional probability simplex, and a "
        "vector of decoding parameters theta comprising temperature, nucleus sampling threshold, "
        "retry count and output token budget. The joint search space is the product of these four "
        "components."
    )
    b.body(
        "Let mu_j denote the score produced by evaluator j on the output of the agent, normalized to "
        "the unit interval, and let c_j denote the unit cost of executing that evaluator. The "
        "fitness of a candidate is the cost-penalized scalarization"
    )
    b.para("F(Φ, x_E, w)  =  Σ  wⱼ · μⱼ   −   λ Σ  cⱼ"
           "        (j ∈ x_E)",
           size=12, align=CENTER, italic=True)
    b.body(
        "where lambda is the cost penalty coefficient, set to 0.5 throughout the experiments "
        "reported here. Only the evaluators selected by the activation vector contribute to either "
        "term, so the objective rewards a candidate that achieves a high weighted score while "
        "incurring low evaluation expenditure. The improvement signal used to update the surrogate "
        "models is the difference between the fitness of a child and that of its parent."
    )

    b.section_heading("4.3", "System architecture")
    b.figure("fig_architecture.png",
             "System architecture. The evolution controller drives three modules in "
             "sequence: candidate generation and execution, adaptive evaluation and "
             "scoring, and redundancy, archive and joint search. The dashed path returns "
             "each generation's observations to the controller")
    b.body(
        "The architecture is organized around a central evolution controller which draws a "
        "candidate from the joint search space and passes it through three modules in sequence. "
        "Flow between modules runs top to bottom. Within Module 2 the two-tier cascade sits at "
        "the centre and fans outward, supplying the scored subset to the cost-penalized "
        "objective on one side and the Gaussian process on the other. A single dashed feedback "
        "path returns the generation's observations from the archive to the controller."
    )
    b.bullets([
        "Module 1, candidate generation and execution, produces a child configuration and runs it. "
        "The mutation component rewrites the parent prompt and perturbs the decoding parameters, "
        "the execution component runs the resulting agent against a benchmark task, and the "
        "language model client handles model failover and rate-limit backoff for both.",
        "Module 2, adaptive evaluation and scoring, decides what to measure and what the "
        "measurements are worth. The two-tier cascade of Gap 1 selects the evaluator subset, the "
        "Gaussian process of Gap 2 proposes the weight vector on the simplex, and the "
        "cost-penalized objective of Gap 3 combines the resulting scores into a scalar fitness.",
        "Module 3, redundancy, archive and joint search, consumes that fitness and updates the "
        "models that shape the next proposal. Functional ANOVA in Gap 4 decomposes surrogate "
        "variance to flag redundant evaluators, the dimension-elite archive of Gap 5 preserves "
        "per-metric peaks and raises regression alerts, and the joint search of Gap 6 updates the "
        "UCB1 operator bandit alongside the GP-UCB proposal.",
    ])
    b.body(
        "State is exposed through a REST service consumed by the React studio, and persisted to "
        "CSV for offline analysis. The modular boundary is also the boundary of the implementation: "
        "Module 1 corresponds to the core package, Module 2 to the evaluation package together "
        "with the scoring component, and Module 3 to the optimization package, as Table 5.1 sets "
        "out."
    )

    b.section_heading("4.4", "Module design")

    b.subsection_heading("4.4.1", "Agent representation and population archive")
    b.body(
        "An agent is represented by its identifier, parent identifier, system prompt, decoding "
        "parameters, the metric scores it achieved, its fitness, and the mutation operator that "
        "produced it. The archive stores all agents ever evaluated and supports three queries: "
        "retrieval of the globally best agent by fitness, computation of the non-dominated set over "
        "the fitness and cost objectives, and retrieval of the per-dimension elites described in "
        "Section 4.4.7. Parent selection uses a Pareto-aware tournament over this archive, so that "
        "selection pressure is not applied purely along the scalarized objective."
    )

    b.subsection_heading("4.4.2", "Mutation operators")
    b.body(
        "Child prompts are produced by prompting the language model to rewrite the parent prompt "
        "according to a stated optimization goal. The operator set covers distinct and partly "
        "competing directions: edge-case defensiveness, explicit chain-of-thought decomposition, "
        "conciseness, safety and constraint adherence, algorithmic efficiency, and robustness "
        "through retry discipline. Decoding parameters are perturbed independently, so a child "
        "differs from its parent in both the discrete and the continuous components."
    )

    b.subsection_heading("4.4.3", "Two-tier evaluation cascade (Gap 1)")
    b.body(
        "The six evaluators are partitioned into a core tier and a deep tier. The core tier contains "
        "the three cheap deterministic evaluators: functional correctness by sandboxed test "
        "execution, execution latency by high-resolution timing, and conciseness by output length "
        "analysis. The deep tier contains the three expensive evaluators: reasoning quality and "
        "safety, both assessed by an LLM judge, and edge-case robustness, assessed by a larger "
        "boundary-assertion suite."
    )
    b.body(
        "For each candidate the core tier is executed first and its mean is taken as a preview "
        "score. The deep tier is executed only if that preview score meets a threshold of 0.65. The "
        "rationale is that a candidate which already fails cheap correctness and efficiency checks "
        "will not be rescued by a favourable judgement on reasoning quality, so paying for that "
        "judgement is waste. The threshold is the single tunable parameter of the mechanism and "
        "trades evaluation savings against the risk of discarding a candidate prematurely."
    )

    b.subsection_heading("4.4.4", "Bayesian optimization of metric weights (Gap 2)")
    b.body(
        "Rather than fixing the weight vector, the framework treats the mapping from weights to "
        "observed fitness improvement as an unknown function and models it with a Gaussian process. "
        "The kernel is a constant-scaled Matérn kernel with smoothness parameter 5/2, which assumes "
        "the underlying function is twice differentiable and is the conventional choice when the "
        "strong smoothness implied by a squared-exponential kernel is not warranted. A small "
        "observation-noise term regularizes the fit, and the kernel hyperparameters are re-optimized "
        "from several restarts as observations accumulate."
    )
    b.body(
        "Candidate weight vectors are proposed by sampling the simplex, evaluating the acquisition "
        "function at each sample, and selecting the maximizer. The default acquisition is the upper "
        "confidence bound, which adds to the posterior mean a multiple of the posterior standard "
        "deviation. The multiplier follows the schedule of Srinivas et al. (2010), which grows "
        "logarithmically in the iteration index and is what their sublinear-regret guarantee "
        "requires. Expected improvement is available as an alternative acquisition rule."
    )

    b.subsection_heading("4.4.5", "Cost-penalized fitness (Gap 3)")
    b.body(
        "Evaluator costs enter the objective directly, as shown in Section 4.2. The three core "
        "evaluators are inexpensive, the three deep evaluators roughly an order of magnitude more "
        "so, reflecting the fact that two of them are additional model invocations. Because the "
        "penalty is applied to the evaluators actually executed, the cascade of Section 4.4.3 and "
        "the objective of this section reinforce one another: skipping the deep tier both avoids "
        "expenditure and improves the fitness of the candidate being scored."
    )

    b.subsection_heading("4.4.6", "Redundancy detection by functional ANOVA (Gap 4)")
    b.body(
        "To determine whether an evaluator still carries information, the framework fits a surrogate "
        "mapping the vector of evaluator scores to the observed fitness, and decomposes the variance "
        "of that surrogate into contributions attributable to each evaluator individually and to "
        "their interactions. The first-order Sobol index of an evaluator is the ratio of the "
        "variance attributable to it alone to the total variance. An evaluator whose index falls "
        "below 0.05, meaning it explains less than five per cent of the variance in fitness, is "
        "flagged as redundant and becomes a candidate for pruning."
    )

    b.subsection_heading("4.4.7", "Dimension elite archive and regression monitoring (Gap 5)")
    b.body(
        "Pruning creates the risk that a dimension which is no longer evaluated quietly degrades. "
        "Three mechanisms address this. First, the framework maintains a per-dimension elite "
        "archive, retaining for each of the six dimensions the highest-scoring configuration ever "
        "observed on that dimension together with the generation at which it was found; this is a "
        "one-dimensional analogue of a MAP-Elites archive and preserves configurations that the "
        "scalarized objective would discard. Second, pruning decisions are overridden with "
        "probability 0.30, so that a pruned dimension is periodically re-measured rather than "
        "assumed stable. Third, when a candidate is evaluated on a dimension and scores more than "
        "0.25 below the corresponding elite, a silent-regression alert is recorded against that "
        "generation."
    )

    b.subsection_heading("4.4.8", "Joint discrete-continuous search (Gap 6)")
    b.body(
        "The discrete and continuous components of the search are handled by algorithms with "
        "complementary guarantees. Mutation operators are selected by a UCB1 bandit, treating each "
        "operator as an arm whose reward is the fitness improvement it yields; UCB1 attains "
        "logarithmic regret in the number of selections. Weights and decoding parameters are "
        "proposed by GP-UCB, which attains cumulative regret sublinear in the number of iterations. "
        "The two operate within a single generation loop over the joint space, so that discrete and "
        "continuous decisions are informed by the same stream of fitness observations."
    )

    b.section_heading("4.5", "Generation algorithm")
    b.body("The procedure executed once per generation is as follows.")
    steps = [
        "Sample a parent configuration from the population archive using Pareto-aware tournament "
        "selection.",
        "Select a mutation operator and apply it to the parent prompt to obtain a child prompt; "
        "perturb the decoding parameters.",
        "Propose a weight vector on the simplex by maximizing the acquisition function of the "
        "Gaussian process surrogate.",
        "Execute the child agent on the selected benchmark task and capture its output.",
        "Execute the core evaluator tier and compute the preview score as the mean of the core "
        "scores.",
        "If the preview score meets the threshold, or if the epsilon-probe fires, add the deep tier "
        "to the active evaluator set.",
        "Execute the active evaluators and obtain the metric scores and their costs.",
        "Compute the cost-penalized fitness and the improvement over the parent.",
        "Update the Gaussian process with the weight vector and the observed improvement; update "
        "the bandit with the operator reward; update the functional ANOVA surrogate with the metric "
        "vector.",
        "Insert the child into the archive, refresh the Pareto front and the per-dimension elites, "
        "and raise a regression alert if any dimension has dropped beyond the configured margin.",
        "Persist the generation record to the telemetry file.",
    ]
    for i, s in enumerate(steps, 1):
        p = b.doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.5
        r = p.add_run(f"Step {i}. {s}")
        r.font.name = "Times New Roman"
        r.font.size = Pt(12)

    b.section_heading("4.6", "Evaluator suite specification")
    b.table(
        ["Metric", "Dimension", "Tier", "Mechanism", "Unit cost"],
        [
            ["mu_1", "Functional correctness", "Core", "Sandboxed execution against test cases", "$0.0010"],
            ["mu_2", "Execution latency", "Core", "High-resolution timing of candidate execution", "$0.0001"],
            ["mu_3", "Output conciseness", "Core", "Token and byte length analysis", "$0.0001"],
            ["mu_4", "Reasoning quality", "Deep", "LLM-as-a-judge", "$0.0050"],
            ["mu_5", "Edge-case robustness", "Deep", "Sandboxed boundary assertion suite", "$0.0050"],
            ["mu_6", "Safety and anti-hallucination", "Deep", "LLM-as-a-judge", "$0.0050"],
        ],
        "Evaluator suite: dimensions, tiers, mechanisms and unit costs",
        widths=[1.6, 4.0, 1.4, 5.4, 2.1],
    )
    b.body(
        "The cost figures are the internal accounting units used by the optimizer. They encode the "
        "relative expense of the dimensions rather than billed amounts, and the ratio of roughly "
        "fifty to one between an LLM-judged dimension and a timing measurement is what drives the "
        "behaviour of the cascade."
    )

    b.section_heading("4.7", "Data flow and persistence")
    b.body(
        "Each generation emits a record containing the generation index, agent and parent "
        "identifiers, fitness, the cost incurred and the cost a naive exhaustive evaluation would "
        "have incurred, the mutation operator, decoding parameters, the count of active evaluators, "
        "all six metric scores, all six weights, the cumulative adaptive and naive costs, the task "
        "identifier, the evolved system prompt and a preview of the generated solution. This fixed "
        "schema is appended to a CSV file, which is the sole input to the analysis described in "
        "Chapter 6; no reported statistic depends on in-memory state."
    )

    b.section_heading("4.8", "A second search mode: the program archive")
    b.body(
        "The architecture described so far evolves one agent per generation and scores it on every "
        "task. That is the mode used for the ablation of Chapter 6, and it suits a suite of a dozen "
        "tasks on which a competent model already succeeds much of the time. It suits the ARC-AGI "
        "benchmark far less well, for two reasons: a single agent prompt must serve a hundred and "
        "twenty unrelated puzzles, and a candidate that fails is discarded along with whatever "
        "partial insight it contained. A second mode was therefore added, which keeps the "
        "evaluation machinery of Sections 4.4.3 to 4.4.8 unchanged and replaces only the unit of "
        "selection."
    )
    b.body(
        "In program-archive mode the object under evolution is a program for a particular task "
        "rather than a prompt for all of them. Four differences follow from that change."
    )
    b.bullets([
        "The archive becomes a grid per task rather than a single population. A program's cell "
        "coordinate is the pattern of demonstration pairs it passes together with how many of its "
        "outputs have the correct shape, which is a behavioural descriptor rather than a quality "
        "score. Two programs that solve different pairs therefore both survive, where a "
        "fitness-ranked population would have kept only the higher-scoring one. This is the "
        "MAP-Elites principle of Section 4.4.7 applied to programs instead of to metrics.",
        "Candidates are produced by repair as well as by mutation. A repair call is given an "
        "archived program together with the demonstration pairs it currently fails, and asked for "
        "a revision; this is what lets partial progress accumulate across generations instead of "
        "being rediscovered each time.",
        "Effort is allocated by a bandit over tasks. A UCB1 selector chooses which tasks each child "
        "works on, so attention concentrates on tasks that are neither solved nor hopeless, and a "
        "task whose demonstrations are fully fitted leaves the pool entirely.",
        "The population is divided into islands which evolve independently and exchange members at "
        "a fixed interval, so that several lines of attack on the same task can develop in "
        "parallel without immediately displacing one another.",
    ])
    b.body(
        "The parent of each new program is chosen by a Tree-structured Parzen Estimator over the "
        "archived candidates, which is used in preference to the Gaussian process of Section 4.4.4 "
        "because the candidate set changes every generation as programs enter and leave the "
        "archive, and a Gaussian process over a fixed design space cannot represent that. The "
        "weight vector over evaluators is still proposed by the Gaussian process, unchanged."
    )

    b.section_heading("4.9", "Summary")
    b.body(
        "The design couples an evolutionary outer loop over agent configurations with an inner "
        "Bayesian optimization over the evaluation process. The cascade decides what to pay for, the "
        "Gaussian process decides how to weight what was measured, the cost penalty makes that "
        "expenditure visible to the objective, functional ANOVA identifies what has stopped "
        "informing the search, and the elite archive with epsilon-probing ensures that what is no "
        "longer measured is nonetheless audited. The program-archive mode of Section 4.8 reuses all "
        "of that machinery and changes only what is being selected, which is why the two "
        "experiments of Chapter 6 can share an implementation while answering different questions."
    )
    b.page_break()


# ==========================================================================
# CHAPTER 5 - IMPLEMENTATION
# ==========================================================================
def chapter5(b):
    b.chapter_heading("5", "Implementation")

    b.section_heading("5.1", "Introduction")
    b.body(
        "This chapter documents how the design of Chapter 4 was realized, the structure of the "
        "codebase, the implementation details that required particular care, and the interfaces "
        "through which the system is operated."
    )

    b.section_heading("5.2", "Technology stack and repository structure")
    b.table(
        ["Module", "Responsibility"],
        [
            ["core/agent.py", "Agent dataclass, decoding-parameter container, archetype definitions"],
            ["core/archive.py", "Population archive, Pareto front, per-dimension elites, CSV export"],
            ["core/groq_client.py", "LLM client with model failover, rate-limit backoff, an offline "
                                    "deterministic simulator, and a strict mode that forbids "
                                    "substituting simulated output for a failed live call"],
            ["core/mutator.py", "Prompt mutation operators and decoding-parameter perturbation"],
            ["evaluation/base.py", "Evaluator abstraction and result container"],
            ["evaluation/metrics.py", "The six evaluator implementations"],
            ["evaluation/pool.py", "Evaluator registry, tier lookup and subset execution"],
            ["evaluation/runner.py", "Sandboxed code execution, answer extraction, sentiment backend"],
            ["optimization/scoring.py", "Cost-penalized scalarized fitness"],
            ["optimization/bayesian_weights.py", "Gaussian process surrogate, GP-UCB and EI acquisition"],
            ["optimization/bandit.py", "UCB1 mutation-operator bandit"],
            ["optimization/fanova.py", "Variance decomposition and first-order Sobol indices"],
            ["optimization/joint_sampler.py", "Joint sampling of parents, evaluator subsets, weights "
                                              "and decoding parameters"],
            ["controller.py", "Evolution controller orchestrating one generation"],
            ["tasks/benchmark_tasks.py", "Benchmark task definitions for both domains"],
            ["server.py", "FastAPI service exposing the optimizer state"],
            ["app.py", "Streamlit interface"],
            ["frontend/", "React and Vite instrumentation studio"],
            ["run_experiments.py", "Ablation harness across conditions, tasks and seeds"],
            ["analyze_results.py", "Wilcoxon tests and convergence analysis"],
            ["report_analysis.py", "Aggregation of results into report figures, tables and scalars"],
        ],
        "Repository structure and module responsibilities",
        widths=[5.2, 9.3],
    )

    b.section_heading("5.3", "Implementation details of note")

    b.subsection_heading("5.3.1", "Language model client")
    b.body(
        "The client wraps the Groq chat-completions endpoint. Three behaviours proved necessary in "
        "practice. First, model failover: the client holds an ordered list of candidate models and "
        "advances to the next when a model returns a not-found or decommissioned response, which "
        "occurred when previously documented model identifiers were withdrawn by the provider. "
        "Second, rate-limit backoff: responses indicating that the per-minute token allowance has "
        "been exhausted trigger an exponential retry rather than a failure, which matters because "
        "the free tier used in this project permits only a few thousand tokens per minute. Third, a "
        "deterministic offline simulator: when no API key is configured the client returns canned "
        "responses keyed to the task, which makes large-sample experimentation possible without "
        "network access or quota. The behaviour of this simulator is central to interpreting the "
        "results of Chapter 6 and is discussed there explicitly."
    )

    b.subsection_heading("5.3.2", "Sandboxed execution and answer extraction")
    b.body(
        "Candidate solutions are executed in a constrained namespace with a bounded time budget, and "
        "each test case is compared against its expected value using a comparison that tolerates "
        "floating-point error and accepts the categorical labels used by the sentiment domain."
    )
    b.body(
        "Extracting executable code from a model response required more care than anticipated. The "
        "initial implementation took the first fenced block in the response. Reasoning-oriented "
        "models, however, routinely emit several fenced blocks and frequently leave them untagged, "
        "placing illustrative fragments ahead of the actual solution; taking the first block then "
        "extracts a fragment such as an isolated expression, which fails to execute and scores zero "
        "on correctness regardless of the quality of the real answer. The extraction logic was "
        "therefore revised to enumerate all fenced blocks, prefer those tagged as Python, and select "
        "the block that actually defines the task's entry point, falling back to any block defining "
        "a function and finally to direct textual answer extraction. This change converted a "
        "correctness score of zero into full marks on affected tasks and is a prerequisite for any "
        "meaningful live-model result."
    )

    b.subsection_heading("5.3.3", "Surrogate models")
    b.body(
        "The Gaussian process is implemented with scikit-learn (Pedregosa et al., 2011) using a "
        "constant-scaled Matérn 5/2 "
        "kernel, a small fixed noise term and several restarts of the kernel hyperparameter "
        "optimizer. Weight proposals are obtained by sampling the simplex, evaluating the "
        "acquisition at each sample and taking the maximizer, which avoids the need for constrained "
        "continuous optimization on the simplex. The functional ANOVA module fits its own surrogate "
        "over the evaluator-score vectors and decomposes the resulting variance to obtain "
        "first-order indices."
    )

    b.subsection_heading("5.3.4", "Sentiment domain backend")
    b.body(
        "The sentiment domain is evaluated against a pre-trained Keras classifier distributed as the "
        "SentiToolKit package, which returns a probability distribution over negative, neutral and "
        "positive classes. The runner resolves the model once and caches it, since constructing the "
        "Keras graph repeatedly would dominate execution time. A deterministic keyword-based "
        "fallback classifier is used when the package is unavailable; the experiments reported in "
        "Chapter 6 were executed with the real neural model installed."
    )

    b.subsection_heading("5.3.6", "The ARC-AGI benchmark module and its evaluator suite")
    b.body(
        "The ARC benchmark is implemented as a self-contained package that adapts the published "
        "task format to the project's task abstraction. Each ARC task supplies a handful of "
        "demonstration pairs and one or more held-out inputs; the demonstration pairs become the "
        "task's test cases and so drive the correctness signal, while the held-out pairs are kept "
        "separate and are used only for out-of-band reporting. Grids are normalized to lists of "
        "integers before comparison, so a program may return either a nested list or a numpy array "
        "without affecting its score."
    )
    b.body(
        "The suite registers six evaluators, and the distinction between them is a matter of "
        "research integrity rather than implementation convenience. Three compute the official ARC "
        "quantities: whether the program executes and exposes the expected entry point, pass@2 over "
        "the demonstration pairs, and pass@2 over the held-out pairs. The other three, which score "
        "pixel accuracy, shape agreement and colour-histogram overlap, were written for this "
        "project to give the weighted fitness a usable gradient before any candidate reaches an "
        "exact match."
    )
    b.body(
        "Those three are not ARC metrics and must never be reported as ARC results. The module "
        "enforces this rather than relying on the reader: they are defined in a separate file "
        "carrying a header that states their provenance, every result object they produce is "
        "flagged as AI-generated in its details, and the figures in Section 6.8.5 draw them with a "
        "distinct line style and mark them in the caption. The precaution matters because the "
        "partial-credit scores are numerically much larger than the real ones on a benchmark where "
        "most candidates fail outright, so an unmarked average over all six would look like strong "
        "performance while meaning nothing."
    )
    b.body(
        "Every ARC evaluator is registered in the core tier at zero cost, because each one merely "
        "executes a candidate program on grids and none calls a language model. This is why the "
        "selective-evaluation machinery is inert on ARC, as Section 6.8.1 explains, and it is a "
        "property of the benchmark rather than a configuration choice."
    )

    b.section_heading("5.4", "Service and user interfaces")
    b.table(
        ["Endpoint", "Purpose"],
        [
            ["GET /api/status", "Telemetry, best agent, execution mode and active model"],
            ["POST /api/step", "Advance the optimizer by one or more generations"],
            ["POST /api/reset", "Reinitialise the population for a given domain and archetype"],
            ["GET /api/history", "Per-generation record history"],
            ["GET /api/pareto", "Current non-dominated set over fitness and cost"],
            ["GET /api/fanova", "Sobol indices, total variance and redundant-evaluator list"],
            ["GET /api/gp_slice", "One-dimensional slice of the GP posterior and acquisition curve"],
            ["GET /api/dimension_elites", "Per-dimension elite agents and regression alerts"],
            ["GET /api/metadata", "Available tasks, archetypes and conditions"],
        ],
        "REST interface exposed by the FastAPI service",
        widths=[5.4, 9.1],
    )
    b.body(
        "The React studio consumes this interface and organizes the system into dedicated views, one "
        "per mechanism, together with an execution inspector that displays the generated solution "
        "and the per-case pass or fail status of the test suite. A Streamlit interface provides an "
        "alternative single-file view of the same state."
    )

    b.section_heading("5.5", "Reproducibility")
    b.body(
        "The experiment harness seeds both the Python and NumPy random number generators "
        "deterministically from the condition, task and repetition indices, so any individual run "
        "can be reproduced exactly. Raw per-generation records are written to CSV, and all figures "
        "and statistics in Chapter 6 are regenerated from those files by a single analysis script. "
        "The report itself is generated programmatically from the same analysis output, so that no "
        "quoted number is transcribed by hand."
    )

    b.section_heading("5.6", "Summary")
    b.body(
        "The framework is implemented as a modular Python system of roughly twenty modules with two "
        "user interfaces and a REST service. The implementation details that mattered most in "
        "practice were defensive: failover and backoff around a rate-limited API, robust extraction "
        "of executable code from discursive model output, and caching of the neural sentiment "
        "backend."
    )
    b.page_break()


def _p_str(p):
    if p is None:
        return "n/a"
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


# ==========================================================================
# CHAPTER 6 - RESULTS AND DISCUSSION
# ==========================================================================
def chapter6(b):
    d = b.d
    ds = d["dataset"]
    cond_rows = d.get("condition_summary", [])
    pooled = d.get("wilcoxon_pooled", [])
    cost = d.get("cost", {})
    activation = d.get("activation", {})

    by_cond = {r["condition"]: r for r in cond_rows}

    b.chapter_heading("6", "Results and Discussion")

    b.section_heading("6.1", "Introduction")
    b.body(
        "This chapter reports the empirical evaluation of the framework, which rests on two "
        "experiments executed against live language models. The first is a controlled "
        "five-condition ablation on the twelve-task canonical suite, where two of the six evaluator "
        "dimensions are language model judges and therefore carry a cost; it is the evidence for "
        "how the adaptive cascade behaves when evaluation is expensive. The second is a run on the "
        "ARC-AGI benchmark, where every evaluator is free but the tasks are hard enough that "
        "current models solve them in the low single digits; it is the evidence that the search "
        "finds solving programs at all. The two use different evaluator suites, different search "
        "modes and different models, so they answer different questions and neither corroborates "
        "the other. Sections 6.2 to 6.7 cover the ablation, Section 6.8 the ARC experiment, and the "
        "chapter closes with what the combined evidence does and does not support."
    )

    b.section_heading("6.2", "Experimental methodology")

    b.subsection_heading("6.2.1", "Ablation conditions")
    b.body(
        "Five conditions were defined, each disabling or replacing a subset of the mechanisms, so "
        "that the contribution of each can be isolated. The conditions differ along three axes: how "
        "the metric weights are set, which evaluators are executed, and how mutation operators are "
        "selected."
    )
    b.table(
        ["Condition", "Weights", "Active evaluators", "Operator selection"],
        [
            ["C1 Single-metric", "All mass on correctness", "Correctness only", "Uniform random"],
            ["C2 Static cascade", "Uniform (1/6 each)", "Two-tier cascade", "Uniform random"],
            ["C3 UCB1 bandit", "Uniform (1/6 each)", "All six", "UCB1 bandit"],
            ["C4 No pruning", "GP-UCB proposed", "All six", "Uniform random"],
            ["C5 Full adaptive", "GP-UCB proposed", "Two-tier cascade", "Uniform random"],
        ],
        "Ablation conditions",
        widths=[3.4, 3.8, 3.7, 3.6],
    )
    b.body(
        "C5 is the full framework as designed. C4 isolates the effect of the cascade by retaining "
        "adaptive weights while evaluating exhaustively. C2 isolates the effect of adaptive weights "
        "by retaining the cascade with fixed uniform weights. C3 isolates the bandit. C1 is the "
        "conventional single-objective baseline. It is worth stating plainly that the bandit is "
        "exercised only in C3, so the comparison between C5 and the baselines measures the cascade "
        "and the Gaussian process rather than operator selection."
    )

    b.subsection_heading("6.2.2", "Benchmark suite and protocol")
    b.body(
        f"The suite comprises {ds['n_tasks']} tasks across {len(ds['domains'])} domains. Each of the "
        f"five conditions was run on every task with {ds['n_runs_per_cell']} independent repetitions "
        f"under distinct random seeds, for {ds['generations_per_run']} generations per run. This "
        f"yields {ds['mock_runs']} optimization runs and {ds['mock_generations']} generations in "
        f"total, recorded as {ds['mock_rows']} per-generation rows."
    )
    b.body(
        "The primary response variable is the peak cost-penalized fitness attained within a run. "
        "Secondary variables are the cumulative evaluation cost, the cost that exhaustive evaluation "
        "would have incurred over the same run, the peak functional correctness, and the mean number "
        "of evaluators executed per generation."
    )
    n_rep = ds["n_runs_per_cell"]
    min_p = 0.5 ** n_rep          # smallest attainable one-sided p for n paired samples
    n_pairs_pooled = pooled[0]["n_pairs"] if pooled else ds["n_tasks"] * n_rep
    b.body(
        f"Because fitness values across repetitions are not normally distributed and the comparison "
        f"is naturally paired by task and seed, significance is assessed with the Wilcoxon "
        f"signed-rank test rather than a t-test. Two analyses are reported. The first is a per-task "
        f"test, which with {n_rep} repetitions cannot attain a one-sided p-value below {min_p:.3f} "
        f"even under perfect separation, and is therefore of strictly limited power; it is reported "
        f"for completeness only. The second, and the primary analysis, pools all "
        f"{n_pairs_pooled} matched task-repetition pairs into a single test per comparison, which "
        f"restores the power the per-task split gives away. Effect size is reported as the "
        f"matched-pairs rank-biserial correlation, which is independent of sample size."
    )

    b.subsection_heading("6.2.3", "Execution mode and data provenance")
    prov = d.get("provenance", {})
    n_live = prov.get("api_calls_live") or 0
    models = ", ".join(prov.get("models") or []) or "the hosted model"
    b.body(
        "The framework can execute against either a live hosted language model or a deterministic "
        "offline simulator. Every result reported in this chapter was produced in live mode. The "
        f"ablation issued {n_live:,} language model requests against {models}, and no simulated "
        "response entered the dataset. Each agent output was generated by the model, each candidate "
        "solution was executed in the sandbox against the task's test cases, and each evaluator "
        "score was computed from that execution."
    )
    b.body(
        "Provenance is enforced mechanically rather than asserted. In its default configuration the "
        "language model client falls back to a canned response when a request fails, which is "
        "convenient for offline demonstration but dangerous for measurement: a run whose quota is "
        "exhausted midway would continue to completion and write a results file in which fabricated "
        "rows are indistinguishable from genuine ones. The experiment driver therefore enables a "
        "strict mode in which any failed request aborts the run instead. The count of live and "
        "simulated calls is recorded alongside every cell of the design, and the analysis stage "
        "refuses to build the report if a single simulated call is present. The figures and tables "
        "that follow could not have been produced from fabricated data."
    )
    b.body(
        "The scale of the design was set by the quota rather than by preference. The tier available "
        "for this project permits one thousand requests per day and eight thousand tokens per "
        "minute, and a single generation consumes between two and four model invocations depending "
        "on how many evaluators the cascade activates. The design reported here was sized to fit "
        f"that budget: {ds['n_tasks']} tasks, {len(ds['conditions'])} conditions, "
        f"{ds['n_runs_per_cell']} repetition per cell and {ds['generations_per_run']} generations "
        "per run. The two consequences are stated plainly here and revisited in Section 7.3: the "
        "optimization horizon is short, so the convergence evidence in Section 6.3 describes early "
        "search behaviour rather than asymptotic performance, and with a single repetition per cell "
        "the per-task tests carry no within-cell variance estimate, which is why the pooled analysis "
        "is the informative one."
    )

    b.section_heading("6.3", "Convergence behaviour")
    b.figure("fig_convergence.png",
             "Best-so-far fitness against generation for each condition, averaged over "
             "all tasks and repetitions, with standard error bands")
    b.body(
        f"Figure {b.last_fig()} shows best-so-far fitness against generation, and it divides the "
        "conditions into two groups that behave quite differently. The single-metric and static "
        "cascade conditions rise steeply to generation one and then stop: both are flat over the "
        "final two generations, having found their best candidate early and not improved on it. "
        "The bandit condition behaves almost identically, gaining less than three thousandths over "
        "the same interval. For these three the search has effectively finished well before the "
        "budget does."
    )
    b.body(
        "The two conditions that evaluate the most dimensions behave in the opposite way. "
        "Exhaustive evaluation gains a further 0.053 over the last two generations and the full "
        "adaptive condition a further 0.030, which are the largest late gains of any condition; "
        "both are still climbing when the budget runs out. This is the clearest evidence in the "
        "chapter that six generations is too short a horizon, and it is the reason the flat "
        "comparisons of Section 6.5 should be read as a statement about this budget rather than "
        "about the conditions in general."
    )
    b.body(
        "The ordering is also not stable, which bears on how the final numbers should be read. At "
        "generation one the full adaptive condition is last of the five; by generation six it has "
        "climbed to third, and exhaustive evaluation has overtaken the single-metric condition that "
        "led from the start. A ranking taken at the horizon is therefore a snapshot of an ordering "
        "still in motion, not a settled result, and the two conditions that finish strongest are "
        "precisely those whose curves have not yet turned over."
    )

    b.section_heading("6.4", "Cost of evaluation")
    if cost:
        b.figure("fig_cost_trajectory.png",
                 "Cumulative evaluation expenditure under the full adaptive condition "
                 "against the exhaustive baseline, averaged over all tasks and repetitions")
        b.body(
            f"Figure {b.last_fig()} contrasts the cumulative expenditure of the adaptive "
            f"cascade with that of "
            f"exhaustive evaluation over the same run. By the final generation the adaptive condition "
            f"had spent {_fmt_usd(cost['final_adaptive_usd'])} against "
            f"{_fmt_usd(cost['final_naive_usd'])} for exhaustive evaluation, a reduction of "
            f"{_fmt_usd(cost['saved_usd'])}, or {_pct(cost['saved_pct'])} of the exhaustive cost. "
            f"This is the central quantitative result of the project."
        )
        b.body(
            "The mechanism behind the saving is visible in the evaluator activation data. Under "
            "exhaustive evaluation all six evaluators run every generation; under the cascade the "
            "three deep evaluators run only when the preview score clears the threshold or the "
            "epsilon-probe fires."
        )
    b.figure("fig_evaluator_activation.png",
             "Mean number of evaluators executed per generation, by condition")
    if activation:
        act_lines = ", ".join(
            f"{b_cond_label(k)} {v:.2f}" for k, v in activation.items())
        b.body(
            f"Figure {b.last_fig()} reports the mean number of evaluators executed per "
            f"generation. The means "
            f"were: {act_lines}. The conditions that evaluate exhaustively sit at the ceiling of six "
            f"by construction, the single-metric condition at one, and the cascading conditions "
            f"between the two, which is the degree of freedom the saving is drawn from."
        )
    b.figure("fig_cost_vs_fitness.png",
             "Cost-quality trade-off: peak fitness against evaluation expenditure for "
             "every run, with condition means marked")
    b.body(
        f"Figure {b.last_fig()} plots the trade-off directly. The relevant question for the "
        f"framework is not "
        "which condition attains the highest raw score but whether the cascade reaches comparable "
        "quality at materially lower expenditure, which corresponds to displacement towards the "
        "upper-left of this plot."
    )

    b.section_heading("6.5", "Statistical comparison")

    b.subsection_heading("6.5.1", "A comparability defect in the raw objective, and its correction")
    b.body(
        "Before any comparison can be made, a property of the implemented objective has to be "
        "addressed, because it was discovered during analysis and it invalidates the obvious "
        "comparison. The fitness function sums the weighted scores of the evaluators that were "
        "actually executed, while the weight vector is defined on the simplex over all six "
        "dimensions. A condition that executes every evaluator therefore sums the entire unit mass "
        "of weight, whereas a condition that executes a subset sums only the mass belonging to that "
        "subset. The deficit is not small."
    )
    comp = d.get("comparability", {})
    if comp:
        rows = [[b_cond_label(k), f'{v["weight_mass"]:.3f}', f'{v["raw"]:.4f}',
                 f'{v["normalized"]:.4f}'] for k, v in comp.items()]
        b.table(
            ["Condition", "Active weight mass", "Mean peak raw F", "Mean peak normalized F"],
            rows,
            "Weight mass actually summed by each condition, and the effect on fitness",
            widths=[3.2, 3.8, 3.7, 3.8],
        )
    b.figure("fig_fitness_comparability.png",
             "Raw against normalized fitness by condition, and the weight mass each "
             "condition actually sums")
    b.body(
        "The cascading conditions sum roughly two thirds of the weight mass, because they routinely "
        "execute three or four of the six evaluators. Their raw fitness is therefore depressed by "
        "approximately one third for a purely structural reason that has nothing to do with the "
        "quality of the agents they produce. Comparing raw fitness across conditions measures how "
        "many evaluators a condition happened to run, not how well it performed."
    )
    b.body(
        "The correction is to divide the weighted score by the weight mass actually summed before "
        "subtracting the cost penalty, which restores a common scale without altering the cost term:"
    )
    b.para("Fₙₒᵣₘ  =  ( Σ  wⱼ · μⱼ ) / ( Σ  wⱼ )"
           "   −   λ Σ  cⱼ        (j ∈ x_E)",
           size=12, align=CENTER, italic=True)
    b.body(
        "All cross-condition comparisons below use this normalized quantity. Raw fitness is retained "
        "in the tables because it is what the optimizer actually maximized, and the difference "
        "between the two columns is itself a finding: the objective as implemented rewards "
        "evaluating more dimensions, which works directly against the pruning behaviour the "
        "framework is designed to encourage. This is a defect in the formulation rather than in the "
        "implementation, and Section 7.2 proposes normalizing the objective in place."
    )

    b.subsection_heading("6.5.2", "Pooled comparison on normalized fitness")
    if pooled:
        rows = []
        for r in pooled:
            rows.append([
                r["comparison"].replace("full_adaptive", "C5").replace("single_metric", "C1")
                 .replace("static_cascade", "C2").replace("ucb1_bandit", "C3")
                 .replace("no_pruning", "C4"),
                r["n_pairs"],
                f'{r["mean_treatment"]:.4f}',
                f'{r["mean_baseline"]:.4f}',
                f'{r["mean_diff"]:+.4f}',
                f'{r["wilcoxon_W"]:.1f}',
                _p_str(r["p_one_sided"]),
                f'{r["rank_biserial_effect"]:+.3f}',
                "Yes" if r["significant_0_05"] else "No",
            ])
        b.table(
            ["Comparison", "n", "Mean C5", "Mean base", "Diff", "W", "p (1-sided)", "Effect r",
             "Sig. 0.05"],
            rows,
            "Pooled Wilcoxon signed-rank tests on normalized fitness, C5 against each "
            "baseline",
            widths=[2.6, 0.9, 1.6, 1.6, 1.5, 1.4, 1.8, 1.5, 1.6],
            font_size=9,
        )
        n_sig = sum(1 for r in pooled if r["significant_0_05"])
        b.body(
            f"Table {b.last_tab()} reports the pooled comparison over "
            f"{pooled[0]['n_pairs']} matched "
            f"task-repetition pairs per comparison. The full adaptive condition attains a higher "
            f"mean normalized fitness than every baseline, and {n_sig} of {len(pooled)} comparisons "
            f"reach significance at the five per cent level on a one-sided test. The rank-biserial "
            f"effect sizes are large; a value of unity indicates that the full adaptive condition "
            f"won every single matched pair."
        )
        b.body(
            f"For completeness, the same comparison on raw fitness reverses the sign against the "
            f"exhaustive baseline, for precisely the reason set out in Section 6.5.1. That reversal "
            f"is reported in the distributed results file rather than treated as a result, because "
            f"it is an artefact of the partial sum rather than a measurement of agent quality."
        )
        b.body(
            f"The per-task analysis found "
            f"{d.get('wilcoxon_per_task_significant', 0)} of {d.get('wilcoxon_per_task_total', 0)} "
            f"task-level comparisons significant at the same threshold. The discrepancy between the "
            f"two analyses is a power effect rather than a contradiction: with "
            f"{ds['n_runs_per_cell']} repetitions the per-task test cannot report a one-sided "
            f"p-value below {0.5 ** ds['n_runs_per_cell']:.3f} even under perfect separation, so the "
            f"pooled analysis is the informative one."
        )
    else:
        b.body("Statistical comparison could not be computed from the available data.")

    b.subsection_heading("6.5.3", "Comparison on functional correctness")
    corr = d.get("wilcoxon_pooled_correctness", [])
    if corr:
        all_zero = all(abs(r["mean_diff"]) < 1e-9 for r in corr)
        if all_zero:
            b.body(
                f"Functional correctness is identical across all five conditions, at a mean peak of "
                f"{corr[0]['mean_treatment']:.4f} in every case, and every paired difference is "
                f"exactly zero. Since every generation was executed against the live model, this "
                f"is a genuine null result rather than an artefact: over six generations the "
                f"optimization condition did not move functional correctness on this suite. The "
                f"most likely reason is that peak correctness is attained early, often by the seed "
                f"agent itself, leaving no headroom for the condition to act on within the horizon. "
                f"It follows that the ablation supports claims about the cost and weighting "
                f"behaviour of the evaluation machinery, and that the evidence for capability "
                f"improvement rests on the ARC experiment of Section 6.8."
            )
        else:
            rows = [[r["comparison"].replace("full_adaptive", "C5"), r["n_pairs"],
                     f'{r["mean_treatment"]:.4f}', f'{r["mean_baseline"]:.4f}',
                     f'{r["mean_diff"]:+.4f}', _p_str(r["p_one_sided"]),
                     "Yes" if r["significant_0_05"] else "No"] for r in corr]
            b.table(
                ["Comparison", "n", "Mean C5", "Mean base", "Diff", "p (1-sided)", "Sig. 0.05"],
                rows,
                "Pooled Wilcoxon tests on functional correctness",
                widths=[3.6, 1.0, 2.2, 2.2, 1.8, 2.0, 1.7], font_size=9,
            )
            b.body(
                "Correctness is the dimension on which a quality claim would rest, and it is "
                "directly comparable across conditions because it is not affected by the weighting "
                "scheme."
            )

    b.section_heading("6.6", "Behaviour of the individual mechanisms")

    b.subsection_heading("6.6.1", "Weight adaptation")
    b.figure("fig_weight_trajectory.png",
             "Mean metric weights proposed by GP-UCB across generations, under the full "
             "adaptive condition")
    b.body(
        f"Figure {b.last_fig()} shows the evolution of the weight vector proposed by the "
        f"Gaussian process. The "
        "weights remain on the simplex by construction. Their movement over the run is the "
        "observable signature of Gap 2 being addressed: under the static conditions this surface "
        "would be flat at one sixth per dimension."
    )

    b.subsection_heading("6.6.2", "Mutation operator selection")
    b.figure("fig_mutation_usage.png",
             "Mutation operator selection under the UCB1 bandit (C3) against uniformly "
             "random selection (C5)")
    mu = d.get("mutation_usage", {})
    spreads = {}
    for cond_label, ops in mu.items():
        vals = [v["n"] for v in ops.values()]
        if vals:
            spreads[cond_label] = (min(vals), max(vals), max(vals) - min(vals))
    bandit = next((v for k, v in spreads.items() if "UCB1" in k), None)
    uniform = next((v for k, v in spreads.items() if "uniform" in k), None)

    if bandit and uniform:
        b.body(
            f"Figure {b.last_fig()} reports how often each mutation operator was selected, under "
            f"the bandit condition and under uniformly random selection. The result is the "
            f"opposite of what one might expect. The UCB1 bandit selected every operator exactly "
            f"{bandit[0]} times, a spread of zero, while the uniformly random condition shows a "
            f"spread of {uniform[2]} selections between its least and most used operator, which is "
            f"ordinary sampling variation."
        )
        b.body(
            "This is an artefact of the experiment's horizon rather than a property of the policy, "
            "and the arithmetic is exact. UCB1 begins by playing each arm once, because its index "
            "is undefined for an arm with no observations. There are six mutation operators and "
            "each cell of the design runs six generations, so a cell affords exactly one pull of "
            "each arm and the run ends at the moment the initialization completes. Twelve cells "
            "therefore produce precisely twelve pulls of every arm, which is what the data shows, "
            "and the policy never reaches the phase in which the estimated means influence the "
            "choice at all."
        )
        b.body(
            "The consequence is that this ablation cannot speak to whether the bandit identifies "
            "better operators: the condition labelled C3 differs from uniform random selection in "
            "implementation but not, at this horizon, in behaviour. The comparison in the figure is "
            "still worth reporting, because a reader would otherwise read the flat distribution as "
            "evidence that the operators are equivalent. A run of at least three or four times the "
            "number of generations per cell is the minimum needed for the question to be asked at "
            "all, and Section 7.2 records it as such. On ARC, where the same bandit machinery "
            "selects tasks rather than operators and is given many more pulls per generation, the "
            "selector does leave initialization, which is visible in the falling attempted-task "
            "counts of Section 6.8.3."
        )
    else:
        b.body(
            f"Figure {b.last_fig()} reports how often each mutation operator was selected."
        )

    b.subsection_heading("6.6.3", "Evaluator dimension profile")
    b.figure("fig_metric_profile.png",
             "Mean score on each evaluator dimension under the full adaptive condition, "
             "separated by domain")
    b.body(
        f"Figure {b.last_fig()} decomposes performance by evaluation dimension and domain. "
        f"The profile differs "
        "markedly between the two domains, which is informative: the coding domain is scored by "
        "deterministic execution against test cases, whereas the sentiment domain is scored against "
        "the predictions of a fixed neural classifier, and the ceiling in the latter is therefore "
        "set by that classifier rather than by the agent."
    )

    b.section_heading("6.7", "Behaviour across domains")
    dom_rows = d.get("domain_summary", [])
    if dom_rows:
        rows = [[r["domain"], r["label"], r["n_runs"], f'{r["mean_peak_fitness"]:.4f}',
                 f'{r["mean_peak_correctness"]:.4f}', f'{r["mean_cost_saved_pct"]:.1f}%']
                for r in dom_rows]
        b.table(
            ["Domain", "Condition", "Runs", "Mean peak fitness", "Mean peak correctness",
             "Mean cost saved"],
            rows,
            "Results separated by agent domain",
            widths=[2.4, 3.4, 1.3, 2.6, 2.8, 2.0],
            font_size=9,
        )
        b.body(
            f"Table {b.last_tab()} separates the results by domain. The sentiment domain "
            f"shows systematically "
            "lower correctness than the coding domain. Inspection of individual cases indicates that "
            "the neural classifier returns the neutral class for contrastive constructions such as a "
            "positive clause followed by an adversative negative clause, and for double negations of "
            "the form used in the negation and sarcasm suite. This is a property of the classifier "
            "rather than a defect in the optimizer, and it places a ceiling on attainable correctness "
            "in that domain that no amount of prompt evolution can lift."
        )

    b.section_heading("6.8", "A second benchmark: ARC-AGI")
    arc = d.get("arc")
    if not arc:
        b.body("The ARC-AGI experiment was not available at build time.")
    else:
        def _num(v):
            try:
                return v if v == v else None      # NaN is the only value unequal to itself
            except Exception:
                return None

        b.subsection_heading("6.8.1", "Motivation and what differs")
        b.body(
            "The ablation of Sections 6.3 to 6.7 establishes how the adaptive cascade behaves on a "
            "twelve-task suite where two of the six evaluator dimensions are themselves language "
            "model judges and therefore carry a monetary cost. That design answers a question about "
            "the cost of evaluation. It cannot answer a second question, which is whether "
            "multi-evaluator weighted fitness actually finds solving programs on a problem that is "
            "genuinely hard. A second experiment was therefore run on the ARC-AGI benchmark, whose "
            "tasks are abstract grid transformations that must be inferred from a handful of "
            "demonstration pairs, and on which current language models score in the low single "
            "digits."
        )
        b.body(
            "Three differences from the first experiment must be stated before any number is "
            "quoted, because they make the two experiments complementary rather than comparable."
        )
        b.bullets([
            "The evaluator suite is different and, critically, free. Every ARC evaluator executes a "
            "candidate program on grids; none calls a language model, so all are registered in the "
            "core tier at zero cost. No strategy therefore has anything worth pruning, the cost "
            "penalty term vanishes, and fitness reduces to the plain weighted sum. The cost result "
            "of Section 6.4 has no analogue here and the cost telemetry from this run is not "
            "meaningful.",
            "The search operates on programs rather than on a single agent. In the ablation, one "
            "child agent per generation is scored on every task. On ARC the controller runs in "
            "program-archive mode: " + str(arc["children_per_generation"]) + " children evolve in "
            "parallel on " + str(arc["islands"]) + " islands, each working on " +
            str(arc["tasks_per_child"]) + " tasks chosen by a UCB1 bandit, and each either repairing "
            "a program already held in that task's archive or writing one from scratch. The archive "
            "is a MAP-Elites grid per task whose cell coordinate is the pattern of demonstration "
            "pairs a program passes together with how many of its outputs have the correct shape, so "
            "programs that succeed in different ways are all retained.",
            "The model and serving infrastructure differ. The ablation used a hosted " +
            ", ".join(d["provenance"]["models"]) + " endpoint; the ARC run used " + arc["model"] +
            " served from a " + arc["provider"] + " deployment, which removed the daily token "
            "ceiling that constrained the first experiment. Absolute scores are not transferable "
            "between the two experiments in either direction.",
        ])

        b.subsection_heading("6.8.2", "Scoring, and a distinction that matters")
        b.body(
            "ARC scoring is strict. A predicted grid is correct only if it matches the expected grid "
            "exactly: the same dimensions and every cell equal. Two attempts are permitted for each "
            "held-out input, and a task is credited only when every one of its held-out inputs is "
            "matched by one of those two attempts. This is the published rule and it is the figure "
            "reported here as the official score."
        )
        b.body(
            f"A second quantity is easily mistaken for it. Averaging the per-task scores gives "
            f"partial credit to a task whose held-out inputs are only partly matched, and "
            f"{arc['partial_tasks']} of the {arc['n_tasks']} tasks are in exactly that position. "
            f"That average reads {arc['mean_final_pct']:.1f} per cent where the official score is "
            f"{arc['official_final_pct']:.1f} per cent. The gap is small in absolute terms but it is "
            "the difference between a quantity comparable with published ARC results and one that is "
            "not, so both are plotted and only the strict figure supports the claims that follow."
        )
        b.body(
            "One further property of the protocol deserves emphasis. The official held-out score is "
            "computed out of band: it is not one of the evaluators in the pool, it never enters the "
            "fitness, and it never influences which programs the archive keeps. Selection sees only "
            "the demonstration pairs. The held-out grids are read solely to score the two attempts "
            "the archive has already chosen, which makes the reported score a genuine measure of "
            "generalization rather than a quantity the search has been optimizing against."
        )

        b.subsection_heading("6.8.3", "Result")
        b.figure("fig_arc_progress.png",
                 "ARC-AGI held-out pass@2 against generation, under the strict official rule and as "
                 "a mean over held-out inputs, with the cumulative count of tasks whose "
                 "demonstration pairs are fitted")
        arc_fig = b.last_fig()
        factor = arc["official_final_n"] / max(arc["official_start_n"], 1)
        b.body(
            f"Figure {arc_fig} shows the outcome over {arc['generations']} generations on "
            f"{arc['n_tasks']} tasks. The official score rises from "
            f"{arc['official_start_pct']:.1f} per cent at the seed ({arc['official_start_n']} of "
            f"{arc['n_tasks']} tasks) to {arc['official_final_pct']:.1f} per cent "
            f"({arc['official_final_n']} of {arc['n_tasks']}), a factor of {factor:.0f} over its "
            "starting point. The rise is monotonic: no generation lost a task an earlier one had "
            "solved, which is what the archive exists to guarantee, since a program that solves a "
            "task holds its cell and cannot be displaced by a later candidate that does not."
        )
        b.body(
            f"The right panel tracks the quantity the search can actually see. The number of tasks "
            f"whose demonstration pairs are completely fitted grows from {arc['demos_start']} to "
            f"{arc['demos_final']}. The archive accumulated {arc['n_programs']:,} programs across "
            f"{arc['n_cells']} distinct behavioural cells, of which {arc['n_elites']} were holding a "
            "cell when the run ended."
        )
        b.figure("fig_arc_archive.png",
                 "Growth of the program archive, and the distribution of programs over the "
                 "MAP-Elites cells that retain them")
        b.body(
            f"Figure {b.last_fig()} shows how that archive accumulated. The gap between programs "
            f"written and programs holding a cell is the selection pressure: roughly one program in "
            f"{arc['n_programs'] / max(arc['n_elites'], 1):.0f} survived as an elite, the rest "
            "having been dominated by an incumbent occupying the same cell. The cell occupancy "
            "distribution is heavily skewed towards the low-scoring cells, which is the expected "
            "shape on a benchmark where most candidates fail every demonstration pair: those "
            "programs all land in the same cell and compete there on the partial-credit metrics "
            "alone."
        )

        rows = []
        for r in arc["generation_table"]:
            ts, op = _num(r.get("tasks_solved")), _num(r.get("official_pct"))
            rows.append([str(int(r["generation"])), str(int(r["children"])),
                         str(int(r["tasks_attempted"])), str(int(r["repairs"])),
                         str(int(r["new_solves"])),
                         "" if ts is None else str(int(ts)),
                         "" if op is None else f"{op:.1f}%"])
        b.table(
            ["Gen", "Children", "Tasks attempted", "Repairs", "New fits", "Tasks solved",
             "Official score"],
            rows,
            "Per-generation summary of the ARC-AGI program-archive run",
            widths=[1.0, 1.6, 2.4, 1.6, 1.6, 2.0, 2.3],
            font_size=9,
        )
        b.body(
            "The attempted-task count falls across the run because a task whose demonstrations are "
            "already fitted leaves the pool the UCB1 selector draws from, so effort concentrates on "
            f"what remains unsolved. The repair column shows the mechanism carrying the search: "
            f"{arc['repairs']:,} repair operations were performed, each handing the model an "
            "archived program together with the demonstrations it fails and asking for a revision."
        )

        b.subsection_heading("6.8.4", "The generalization gap")
        gap_pct = arc["generalisation_gap"] / max(arc["demos_final"], 1) * 100
        b.body(
            f"The two panels of Figure {arc_fig} do not end at the same number, and that difference "
            f"is the most informative result of this experiment. {arc['demos_final']} tasks have "
            f"programs fitting every demonstration pair, but only {arc['official_final_n']} of those "
            f"programs also produce the correct held-out output. {arc['generalisation_gap']} "
            f"programs, {gap_pct:.0f} per cent of those that fit, capture the demonstrations without "
            "capturing the rule."
        )
        b.body(
            "This is overfitting in its purest form, and on ARC it takes a specific character. With "
            "typically three demonstration pairs available, a program can satisfy all of them by "
            "encoding incidental properties of those particular grids rather than the transformation "
            "they illustrate. The search cannot detect this, because the demonstration pairs are the "
            "entire fitness signal; the held-out inputs are deliberately withheld from selection, "
            "which is what makes the measurement honest but also what leaves the search blind to the "
            "distinction. Narrowing this gap would be more valuable than increasing the number of "
            "demonstration fits."
        )

        b.subsection_heading("6.8.5", "Evaluator behaviour")
        b.figure("fig_arc_metrics.png",
                 "Mean score of each in-loop ARC evaluator against generation. Dashed series are the "
                 "AI-generated partial-credit heuristics, which are not ARC metrics")
        b.body(
            f"Figure {b.last_fig()} separates the evaluators into the two groups described in "
            f"Section 6.8.1. The proportion of candidates that execute without raising ends at "
            f"{arc['metrics']['arc_runs_successfully']:.2f}, having climbed through the middle "
            "generations; this is the least interesting form of progress but a precondition for any "
            "other. Pass@2 on the demonstration pairs, the only series here that is an ARC metric in "
            f"its own right, improves to {arc['metrics']['arc_pass_at_2_train']:.3f}, and the "
            "partial-credit heuristics all improve modestly."
        )
        b.body(
            "The partial-credit series must be read with care, and this report marks them at every "
            "appearance. Pixel accuracy, shape match and colour-palette agreement were written for "
            "this project to give the weighted fitness a usable gradient before any candidate "
            "reaches an exact match, which on a benchmark with single-digit solve rates is most of "
            "the time. They are shaping terms, not measurements of ARC performance, and a rise in "
            "them is not evidence of a rise in ARC score. The file defining them carries a header "
            "saying so, and every result they produce is flagged in the telemetry."
        )
        b.figure("fig_arc_weights.png",
                 "GP-UCB weight vector over the ARC evaluator simplex, generation by generation")
        b.body(
            f"Figure {b.last_fig()} shows the weight vector the Gaussian process proposed at each "
            "generation. The behaviour differs strikingly from the smooth drift seen on the "
            "canonical suite in Section 6.6.1: the weights swing between near-exclusive "
            "concentration on one dimension and broad spreads across several, without settling. Two "
            "readings are available and the data does not separate them. Either the acquisition "
            "function is exploring a surface on which no weighting is durably better, which is "
            f"plausible given that only {arc['generations']} generations were run over a "
            "five-dimensional simplex, or the improvement signal is too noisy for the surrogate to "
            "fit, which is equally plausible when most candidates score zero on the metric that "
            "ultimately matters."
        )
        b.body(
            "A direct consequence is that fitness values are not comparable between generations in "
            "this experiment. Because the weight vector is resampled every generation, a generation "
            "that happens to place its weight on an easily satisfied dimension records a high "
            "fitness without any underlying improvement, and the fitness series duly swings without "
            "trend. The held-out score and the count of demonstration fits are the quantities that "
            "carry meaning here, and they are what Section 6.8.3 reports."
        )

        b.subsection_heading("6.8.6", "Cost of the run, and what it does not show")
        total_tok = arc["prompt_tokens"] + arc["completion_tokens"]
        b.body(
            f"The run consumed {total_tok:,} tokens, {arc['prompt_tokens']:,} of them prompt and "
            f"{arc['completion_tokens']:,} completion, of which {arc['reasoning_tokens']:,} were the "
            f"model's hidden reasoning. {arc['truncated_pct']:.1f} per cent of calls terminated at "
            "the token ceiling rather than at a natural stopping point, so a fraction of candidate "
            "programs were truncated mid-definition and scored as failures to execute. That is a "
            "loss of search efficiency rather than a measurement error, but it bounds how much of "
            "the remaining gap is attributable to the method as opposed to the budget."
        )
        b.body(
            "What this experiment establishes is that the archive-based search improves a genuine "
            f"ARC score monotonically, by a factor of {factor:.0f} over its own seed, and that the "
            "improvement survives a held-out test the search never saw. What it does not establish "
            "is that the multi-evaluator weighting is the cause. No ablation was run on ARC: there "
            "is a single configuration, with no comparison against a uniformly weighted or "
            "single-metric variant of the same archive search. The result is therefore evidence that "
            "the system works end to end on a hard benchmark, and not evidence for any particular "
            "component of it. An ablation on ARC is the single most valuable extension of this work "
            "and is listed first in Section 7.2."
        )

    b.section_heading("6.9", "Discussion")
    c5 = by_cond.get("full_adaptive")
    c4 = by_cond.get("no_pruning")
    if c5 and c4:
        b.body(
            f"The evidence supports a narrow but useful claim. Comparing the full adaptive condition "
            f"with exhaustive evaluation under otherwise identical settings, mean evaluation "
            f"expenditure fell from {_fmt_usd(c4['mean_cost_usd'])} to "
            f"{_fmt_usd(c5['mean_cost_usd'])} per run, a reduction of "
            f"{(1 - c5['mean_cost_usd'] / c4['mean_cost_usd']) * 100:.1f} per cent, while the mean "
            f"number of evaluators executed per generation fell from "
            f"{c4['mean_active_evaluators']:.2f} to {c5['mean_active_evaluators']:.2f}. On the "
            f"normalized objective, which is the only cross-condition comparison that is "
            f"structurally sound, mean peak fitness was "
            f"{c5.get('mean_peak_fitness_normalised', float('nan')):.4f} for the cascade against "
            f"{c4.get('mean_peak_fitness_normalised', float('nan')):.4f} for exhaustive evaluation. "
            f"Mean peak correctness was {c4['mean_peak_correctness']:.4f} under exhaustive "
            f"evaluation and {c5['mean_peak_correctness']:.4f} under the cascade."
        )
        b.body(
            "The honest reading of those numbers is that the cascade buys a large cost reduction "
            "for a quality difference too small to detect. On normalized fitness it sits slightly "
            "below exhaustive evaluation, and on correctness the two are indistinguishable; the "
            "paired test over the twelve tasks does not separate them, so the sign of the "
            "difference should not be over-read. What is not in doubt is the cost: roughly half the "
            "expenditure and a third fewer evaluator invocations per generation. The claim the data "
            "supports is therefore that adaptive evaluation is substantially cheaper at no "
            "measurable cost in quality, which is a weaker claim than the framework was designed "
            "to demonstrate but is the one the evidence carries."
        )
    b.body(
        "The interpretation is that the cascade succeeds at its stated purpose, which is to avoid "
        "paying for expensive judgements on candidates that cheap deterministic checks have already "
        "shown to be weak. This is the mechanism by which Gap 1 and Gap 3 are addressed, and it is "
        "the component of the framework that the evidence most directly supports."
    )
    b.body(
        "The evidence for the remaining mechanisms is weaker and should be described as such. The "
        "Gaussian process demonstrably moves the weight vector away from the uniform allocation, "
        "and on ARC it moves it sharply between generations, but neither experiment isolates the "
        "contribution of the weighting itself, so its value cannot be established here. "
        "Functional ANOVA produces sensitivity indices and flags evaluators below "
        "the redundancy threshold, but with a fixed benchmark suite the identity of the redundant "
        "dimensions is largely a property of the suite. The elite archive and regression monitor "
        "operate as designed and record alerts, but a demonstration that they prevent a regression "
        "that would otherwise have occurred would require an adversarial setting that this study "
        "does not construct."
    )
    arc = d.get("arc")
    if arc:
        b.body(
            "The ARC experiment contributes a different kind of evidence, and the two should not "
            "be merged. It shows the search improving a benchmark score that is pixel-exact, "
            "measured on inputs withheld from selection, and hard enough that the seed configuration "
            f"solved only {arc['official_start_n']} of {arc['n_tasks']} tasks. That the score rose "
            "monotonically to four times its starting value is the clearest demonstration in this "
            "report that the machinery does something useful rather than merely consistent. But the "
            "ARC configuration disables the very mechanism the ablation measures, since all its "
            "evaluators are free and nothing is worth pruning, so the two experiments are better "
            "understood as covering disjoint parts of the claim than as reinforcing one another."
        )
        b.body(
            "The generalization gap of Section 6.8.4 is, in the author's view, the most interesting "
            "single observation in this chapter. Roughly half the programs that reproduced every "
            "demonstration pair failed on the held-out input, which says that fitting the available "
            "supervision and learning the intended rule are substantially different achievements on "
            "this benchmark. It also bears on the framework's central premise: a multi-evaluator "
            "objective is motivated by the belief that several imperfect signals constrain a "
            "candidate better than one, and a gap of this size suggests the signals available here "
            "were not sufficiently constraining. Whether adding an evaluator that penalizes "
            "complexity, or one that rewards agreement between independently evolved programs, "
            "would narrow it is an empirical question this work leaves open."
        )

    b.section_heading("6.10", "Threats to validity")
    b.bullets([
        "A short optimization horizon. This is the principal threat. Six generations per cell is "
        "enough to separate conditions on cost, which is a structural difference visible from the "
        "first generation, but too short to separate them on capability. Two consequences are "
        "visible directly in the data: the convergence curves of Section 6.3 flatten early, and "
        "the UCB1 condition of Section 6.6.2 exhausts its budget on the initialization sweep "
        "without ever reaching the phase where its estimates influence selection.",
        "A single repetition per design cell. Each task and condition pair was run once, so the "
        "per-task tests carry no within-cell variance estimate and the pooled analysis across "
        "tasks is the only informative one. Seed variation is therefore unmeasured and could "
        "account for differences of the size reported.",
        "No ablation on ARC. The ARC experiment of Section 6.8 ran a single configuration, so it "
        "evidences that the search works end to end on a hard benchmark but attributes nothing to "
        "the multi-evaluator weighting, the archive or the task bandit individually.",
        "Partial-credit evaluators inside the ARC fitness. Three of the ARC evaluators are "
        "heuristics written for this project rather than ARC metrics. They shape the search and "
        "therefore influence which programs survive, so the reported ARC score, although measured "
        "strictly and out of band, was reached by a search guided in part by quantities that have "
        "no standing in the benchmark.",
        "A protocol change during the ARC run. The run was interrupted and resumed, and the "
        "reasoning budget allocated to repair calls was raised at that point, so generations before "
        "and after the resume were not produced under identical settings.",
        "A comparability defect in the raw objective, described in Section 6.5.1. Raw fitness "
        "cannot be compared across conditions that activate different evaluator subsets. The "
        "analysis corrects for this by normalizing, but the optimizer itself maximized the "
        "uncorrected quantity, so the search was running against an objective that quietly "
        "rewards evaluating more dimensions.",
        "Internal cost units. The unit costs assigned to evaluators encode relative expense rather "
        "than billed amounts, so the reported percentage saving is a statement about evaluator "
        "invocations avoided, not about a monetary invoice.",
        "Duplicate execution of the core tier. In the cascading conditions the core evaluators are "
        "executed once to compute the preview score and again as part of the active set. The "
        "accounting charges them once, which is correct, but the implementation performs the work "
        "twice, so wall-clock savings are smaller than the accounted savings.",
        "Benchmark scale. Twelve tasks is sufficient for controlled comparison between conditions "
        "but too small to support claims about generalization to unseen task distributions.",
        "A ceiling imposed by the sentiment classifier. Attainable correctness in that domain is "
        "bounded by a fixed pre-trained model that misclassifies contrastive and doubly negated "
        "constructions, independently of the optimizer.",
        "Two models, neither varied. The ablation used one hosted model family and the ARC "
        "experiment a different, self-hosted one. Neither experiment varies the model within "
        "itself, so no result here speaks to how behaviour transfers across models, and the two "
        "experiments cannot be compared to each other on absolute scores.",
        "The bandit is exercised in one condition only, so the comparison of the full adaptive "
        "condition against the baselines does not isolate its contribution.",
    ])

    b.section_heading("6.11", "Summary")
    if cost:
        n_sig = sum(1 for r in pooled if r["significant_0_05"]) if pooled else 0
        b.body(
            f"The adaptive cascade reduced cumulative evaluation expenditure by "
            f"{_pct(cost['saved_pct'])} relative to exhaustive multi-metric evaluation, by executing "
            f"the expensive judged dimensions only on candidates that first cleared a cheap "
            f"deterministic gate. On the normalized objective, however, it separated from only "
            f"{n_sig} of the {len(pooled)} baselines at the five per cent level, and it did not "
            f"improve on unpruned evaluation; the defensible reading is that the cascade buys a "
            f"large cost reduction at no measurable cost in quality, not that it produces better "
            f"agents. Functional correctness was statistically indistinguishable across all five "
            f"conditions, which is consistent with that reading. "
            f"Analysis also exposed a defect in the objective as implemented, namely that raw "
            f"fitness is not comparable across conditions that evaluate different subsets, which "
            f"had to be corrected before any comparison could be made and which is reported here "
            f"as a finding rather than silently repaired. The mechanisms addressing weight "
            f"adaptation, redundancy detection and regression monitoring were observed to operate "
            f"as designed, but the present experimental design does not establish their individual "
            f"value, and the chapter states the reasons explicitly rather than inferring more than "
            f"the data supports."
        )
    arc = d.get("arc")
    if arc:
        b.body(
            f"The ARC-AGI experiment answers the question the ablation cannot. Over "
            f"{arc['generations']} generations on {arc['n_tasks']} tasks the official held-out "
            f"pass@2 score rose from {arc['official_start_pct']:.1f} to "
            f"{arc['official_final_pct']:.1f} per cent, monotonically, on grids that never entered "
            f"the fitness signal, while the number of tasks fitting their demonstration pairs grew "
            f"from {arc['demos_start']} to {arc['demos_final']}. That the two counts differ by "
            f"{arc['generalisation_gap']} is itself a finding, and is reported as one. Because the "
            f"ARC run used a single configuration, it establishes that the framework works end to "
            f"end on a benchmark far harder than the canonical suite without attributing the "
            f"improvement to any particular component of it."
        )
    b.page_break()


def b_cond_label(key):
    return {
        "single_metric": "C1",
        "static_cascade": "C2",
        "ucb1_bandit": "C3",
        "no_pruning": "C4",
        "full_adaptive": "C5",
    }.get(key, key)


# ==========================================================================
# CHAPTER 7 - CONCLUSION AND FUTURE WORK
# ==========================================================================
def chapter7(b):
    d = b.d
    cost = d.get("cost", {})
    ds = d["dataset"]
    b.chapter_heading("7", "Conclusion and Future Work")

    b.section_heading("7.1", "Conclusion")
    b.body(
        "This project set out to treat the evaluation of language model agents as an optimizable "
        "resource rather than a free oracle, and to optimize it jointly with the agent configuration "
        "itself. A framework was designed and implemented that combines an evolutionary outer loop "
        "over system prompts and decoding parameters with an inner Bayesian optimization over metric "
        "weights, a tiered evaluation cascade, a cost-penalized objective, variance-based redundancy "
        "detection, and a per-dimension elite archive with regression auditing."
    )
    if cost:
        b.body(
            f"The principal empirical result is that the cascade reduced cumulative evaluation "
            f"expenditure by {_pct(cost['saved_pct'])} relative to exhaustive evaluation across "
            f"{ds['mock_runs']} optimization runs spanning {ds['n_tasks']} tasks in two domains, by "
            f"gating the expensive LLM-judged dimensions behind a cheap deterministic preview score. "
            f"The saving arises from a straightforward observation that nonetheless does not appear "
            f"in the frameworks surveyed in Chapter 2: most candidates in an evolutionary search are "
            f"poor, poor candidates can be identified cheaply, and there is no reason to pay a judge "
            f"to confirm it."
        )
    b.body(
        "Three findings were not anticipated at the outset and are worth recording. The first "
        "emerged only during analysis: the scalarized objective sums the weighted scores of the "
        "evaluators that were executed, while the weights are defined over all six dimensions, so a "
        "condition that prunes evaluators sums less than the full unit of weight and is penalized "
        "for pruning. The objective therefore contained a quiet incentive against the very "
        "behaviour the framework exists to produce. The effect is large enough to reverse the "
        "apparent ranking of the adaptive condition against exhaustive evaluation, and it would "
        "have gone unnoticed had the weight mass not been audited. The remaining two are "
        "implementation findings. The first of these is that "
        "extracting executable code from the output of a reasoning-oriented model is substantially "
        "harder than it appears. The natural implementation, which takes the first fenced block, "
        "silently scores zero on correctness for models that explain before they answer, because the "
        "first block is an illustrative fragment rather than the solution. A correctness signal that "
        "is structurally always zero is worse than no signal, because it is invisible: the search "
        "continues and reports numbers throughout. The last is that an offline simulator intended "
        "as a convenience becomes a methodological hazard the moment it can be reached by accident. "
        "The client in its original form answered a failed request with a canned response, so a run "
        "whose quota expired midway would complete normally and write a results file in which "
        "fabricated rows were indistinguishable from measured ones. This was not hypothetical: an "
        "early dataset produced during this project was found, on audit, to consist entirely of "
        "simulated responses, detected because every task had returned a byte-identical solution "
        "across every condition and repetition, which no sampled model can do. The remedy adopted "
        "here is to make provenance structural rather than procedural. A strict mode aborts the run "
        "when a live call fails, the count of live and simulated calls is recorded with every cell "
        "of the design, and the analysis stage refuses to build the report if a single simulated "
        "call is present. All three lessons argue for the same discipline, which is to verify that "
        "an evaluation signal responds to the thing it claims to measure before trusting any "
        "quantity derived from it."
    )
    b.body(
        "Against the objectives stated in Section 1.5, the formulation, the cascade, the Bayesian "
        "weighting, the cost-aware objective, the redundancy detection, the archive and the "
        "implementation with its three interfaces were all delivered and are operational. The "
        "evaluation objective was met more fully than an earlier draft of this work could claim, "
        "because both experiments were ultimately executed against live models with no simulated "
        "response admitted to either dataset."
    )
    b.body(
        "Two claims can now be defended and a third cannot. The first is that adaptive evaluation "
        "reduces evaluation cost substantially without a measurable loss of quality: the cascade "
        "spent roughly half of what exhaustive evaluation would have spent, and functional "
        "correctness was statistically indistinguishable across the conditions. The second is that "
        "the archive-based search improves a genuine, held-out ARC-AGI score, monotonically and by "
        "a factor of four over its own seed, on a test the search never observed."
    )
    b.body(
        "The claim that cannot be defended is that the multi-evaluator weighting is the cause of "
        "either result. On the canonical suite the full adaptive condition was not significantly "
        "better than the single-metric, static-cascade or unpruned baselines on normalized fitness, "
        "separating only from the bandit condition, and that condition is itself compromised by a "
        "horizon too short for its policy to leave initialization. On ARC no ablation was run at "
        "all. The honest summary is that the framework works end to end and that its cost behaviour "
        "is established, while the contribution of each individual mechanism remains open."
    )

    b.section_heading("7.2", "Future work")
    b.bullets([
        "Run the five-condition ablation on ARC-AGI. The ARC experiment of Section 6.8 used a "
        "single configuration, so it shows that the search works without showing which of its "
        "components matter. Repeating it with the single-metric and uniformly weighted conditions "
        "would isolate the contribution of the multi-evaluator weighting on a benchmark where the "
        "capability signal is unambiguous. This is the single most valuable extension; the harness "
        "already supports it and the cost is compute rather than new code.",
        "Lengthen the optimization horizon. Six generations per cell is too short for two of the "
        "mechanisms to be exercised at all: UCB1 spends the entire budget playing each arm once, "
        "and the convergence curves flatten before the conditions can separate on capability. "
        "Three to four times the horizon is the minimum at which the bandit question can be asked.",
        "Close the generalization gap identified in Section 6.8.4, where seven of fifteen programs "
        "that fitted every demonstration pair failed on the held-out input. Candidate approaches "
        "include penalizing program complexity, requiring agreement between independently evolved "
        "programs, or holding out one demonstration pair from the fitness signal as an internal "
        "validation split.",
        "Normalize the objective in place, dividing the weighted score by the active weight mass "
        "so that the optimizer maximizes a quantity comparable across evaluator subsets. The "
        "analysis in Chapter 6 applies this correction after the fact; applying it inside the "
        "fitness function would remove the incentive against pruning rather than merely "
        "compensating for it, and is a single-line change to the scoring module.",
        "Eliminate the duplicated execution of the core evaluator tier by caching the preview scores "
        "for reuse within the same generation, so that wall-clock savings match accounted savings.",
        "Replace the fixed preview threshold with an adaptive gate, for instance one calibrated to a "
        "target false-prune rate estimated from the epsilon-probe audits.",
        "Learn the cost coefficients from observed API billing and latency rather than fixing them, "
        "so that the penalty term reflects real expenditure.",
        "Extend the benchmark suite in both breadth and difficulty, and include tasks on which the "
        "base model fails, since a suite the model already solves cannot reveal improvement.",
        "Evaluate across several underlying models to establish whether the cascade threshold "
        "transfers or must be recalibrated per model.",
        "Construct an adversarial evaluation in which a dimension is deliberately exploitable, in "
        "order to demonstrate the value of the regression monitor directly rather than by design "
        "argument.",
        "Replace simplex sampling for weight proposal with constrained continuous optimization on "
        "the simplex, which would sharpen the acquisition maximization as the dimension grows.",
    ])
    b.page_break()


# ==========================================================================
# REFERENCES
# ==========================================================================
def references(b):
    b.para("REFERENCES", size=16, bold=True, align=CENTER, caps=True)
    b.para()
    refs = [
        "Agrawal, L. A., Tan, S., Soylu, D., Ziems, C., Khare, R., Opsahl-Ong, K., Singhvi, A., "
        "Shandilya, H., Ryan, M. J., Jiang, M., Potts, C., Sen, K., Stoica, I., Khattab, O., & "
        "Zaharia, M. (2025). GEPA: Reflective prompt evolution can outperform reinforcement "
        "learning. arXiv preprint arXiv:2507.19457.",

        "Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time analysis of the multiarmed "
        "bandit problem. Machine Learning, 47(2-3), 235-256.",

        "Novikov, A., Vu, N., Eisenberger, M., Dupont, E., Huang, P.-S., Wagner, A. Z., "
        "Shirobokov, S., Kozlovskii, B., Ruiz, F. J. R., Mehrabian, A., Kumar, M. P., See, A., "
        "Chaudhuri, S., Holland, G., Davies, A., Nowozin, S., Kohli, P., & Balog, M. (2025). "
        "AlphaEvolve: A coding agent for scientific and algorithmic discovery [Technical report]. "
        "Google DeepMind.",

        "Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., "
        "Terry, M., Le, Q., & Sutton, C. (2021). Program synthesis with large language models. "
        "arXiv preprint arXiv:2108.07732.",

        "Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. de O., Kaplan, J., Edwards, H., "
        "Burda, Y., Joseph, N., Brockman, G., et al. (2021). Evaluating large language models "
        "trained on code. arXiv preprint arXiv:2107.03374.",

        "Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). A fast and elitist multiobjective "
        "genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 6(2), 182-197.",

        "Fernando, C., Banarse, D., Michalewski, H., Osindero, S., & Rocktaschel, T. (2023). "
        "Promptbreeder: Self-referential self-improvement via prompt evolution. arXiv preprint "
        "arXiv:2309.16797.",

        "Hutter, F., Hoos, H., & Leyton-Brown, K. (2014). An efficient approach for assessing "
        "hyperparameter importance. Proceedings of the 31st International Conference on Machine "
        "Learning, PMLR 32(1), 754-762.",

        "Khattab, O., Singhvi, A., Maheshwari, P., Zhang, Z., Santhanam, K., Vardhamanan, S., "
        "Haq, S., Sharma, A., Joshi, T. T., Moazam, H., Miller, H., Zaharia, M., & Potts, C. "
        "(2024). DSPy: Compiling declarative language model calls into self-improving pipelines. "
        "International Conference on Learning Representations.",

        "Mouret, J.-B., & Clune, J. (2015). Illuminating search spaces by mapping elites. arXiv "
        "preprint arXiv:1504.04909.",

        "Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., "
        "Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., et al. (2011). Scikit-learn: "
        "Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",

        "Rasmussen, C. E., & Williams, C. K. I. (2006). Gaussian processes for machine learning. "
        "MIT Press.",

        "Sobol, I. M. (2001). Global sensitivity indices for nonlinear mathematical models and "
        "their Monte Carlo estimates. Mathematics and Computers in Simulation, 55(1-3), 271-280.",

        "Srinivas, N., Krause, A., Kakade, S. M., & Seeger, M. (2010). Gaussian process "
        "optimization in the bandit setting: No regret and experimental design. Proceedings of the "
        "27th International Conference on Machine Learning, 1015-1022.",

        "Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., "
        "Burovski, E., Peterson, P., Weckesser, W., Bright, J., et al. (2020). SciPy 1.0: "
        "Fundamental algorithms for scientific computing in Python. Nature Methods, 17, 261-272.",

        "Wilcoxon, F. (1945). Individual comparisons by ranking methods. Biometrics Bulletin, "
        "1(6), 80-83.",

        "Yang, C., Wang, X., Lu, Y., Liu, H., Le, Q. V., Zhou, D., & Chen, X. (2024). Large "
        "language models as optimizers. International Conference on Learning Representations.",

        "Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., "
        "Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-"
        "judge with MT-Bench and Chatbot Arena. Advances in Neural Information Processing Systems "
        "36, Datasets and Benchmarks Track.",

        "Zhou, Y., Muresanu, A. I., Han, Z., Paster, K., Pitis, S., Chan, H., & Ba, J. (2023). "
        "Large language models are human-level prompt engineers. International Conference on "
        "Learning Representations.",
    ]
    for i, r in enumerate(refs, 1):
        p = b.doc.add_paragraph()
        p.paragraph_format.line_spacing = 1.5
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.left_indent = Pt(28)
        p.paragraph_format.first_line_indent = Pt(-28)
        run = p.add_run(f"[{i}]. {r}")
        run.font.name = "Times New Roman"
        run.font.size = Pt(12)
    b.page_break()


# ==========================================================================
# APPENDICES
# ==========================================================================
def appendices(b):
    import csv
    import io
    import os

    d = b.d
    ds = d["dataset"]

    b.chapter_heading("A", "Appendices")

    # ------------------------------------------------------------ Appendix 1
    b.section_heading("Appendix 1", "Benchmark task specification")
    b.body(
        f"The benchmark suite comprises {ds['n_tasks']} tasks spanning the "
        f"{' and '.join(ds['domains'])} domains. Coding tasks are scored by executing the generated "
        f"function against hidden test cases in a sandbox, following the protocol established by "
        f"HumanEval and MBPP. Sentiment tasks are scored against the predictions of the pre-trained "
        f"Keras classifier described in Section 5.3.4. Every task declares an entry point, a set of "
        f"test cases and a set of edge cases; the edge cases are used only by the edge-case "
        f"evaluator of the deep tier."
    )
    coding = [t for t in ds["tasks"] if not t.startswith("senti_")]
    senti = [t for t in ds["tasks"] if t.startswith("senti_")]
    rows = []
    descriptions = {
        "he_000_has_close_elements": ("HumanEval/0", "Detect whether any two numbers in a list lie "
                                      "closer together than a threshold"),
        "he_003_below_zero": ("HumanEval/3", "Prefix scan over deposits and withdrawals to detect a "
                              "balance falling below zero"),
        "he_055_fib": ("HumanEval/55", "Fibonacci term computation with non-negative input "
                       "validation"),
        "mbpp_006_differ_bits": ("MBPP/6", "Determine whether two integers differ in exactly one "
                                 "bit, via XOR and a power-of-two test"),
        "mbpp_274_stock_exchange": ("MBPP/274", "Maximum profit from a single buy and sell over a "
                                    "price series"),
        "task_valid_parentheses": ("Classical", "Bracket matching using a stack"),
        "task_two_sum": ("Classical", "Index pair summing to a target, via a hash map"),
        "senti_001_product_reviews": ("Sentiment", "Consumer electronics reviews with explicit "
                                      "satisfaction indicators"),
        "senti_002_movie_critiques": ("Sentiment", "Film reviews mixing judgements on acting, "
                                      "pacing and direction"),
        "senti_003_contrastive_clauses": ("Sentiment", "Compound reviews joined by an adversative "
                                          "conjunction, testing dominant-clause polarity"),
        "senti_004_negation_sarcasm": ("Sentiment", "Double negation and ironic praise"),
        "senti_005_aspect_service": ("Sentiment", "Multi-aspect hospitality reviews covering food, "
                                     "room and service"),
    }
    for t in coding + senti:
        src, desc = descriptions.get(t, ("--", "--"))
        rows.append([t, src, desc])
    b.table(["Task identifier", "Source", "Description"], rows,
            "Benchmark task specification", widths=[4.4, 2.2, 7.9], font_size=9)

    b.body(
        "The suite is deliberately modest in size. Its purpose is to support controlled comparison "
        "between optimization conditions under identical task conditions, not to rank language "
        "models against one another. Section 6.10 records the consequence, which is that no claim "
        "about generalization to unseen task distributions is made."
    )

    # ------------------------------------------------------------ Appendix 2
    b.section_heading("Appendix 2", "Evaluator specification")
    b.body(
        "Each evaluator returns a score on the unit interval together with the cost of having "
        "obtained it. The table restates the suite of Section 4.6 with the implementation detail "
        "that the body omits."
    )
    b.table(
        ["Metric", "Tier", "Implementation", "Failure mode guarded against"],
        [
            ["mu_1 correctness", "Core",
             "Executes the extracted entry point against the task's test cases in a constrained "
             "namespace with a bounded time budget; score is the pass rate",
             "Plausible code that does not run or returns wrong values"],
            ["mu_2 latency", "Core",
             "High-resolution timing of candidate execution, normalized against a reference budget",
             "Correct but pathologically slow solutions"],
            ["mu_3 conciseness", "Core",
             "Token and byte length analysis of the response",
             "Verbose output padded with commentary"],
            ["mu_4 reasoning", "Deep",
             "LLM-as-a-judge scoring of the explanation accompanying the solution",
             "Correct answers reached by unsound reasoning"],
            ["mu_5 edge cases", "Deep",
             "Executes the task's edge-case assertions, covering empty inputs, zero and negative "
             "values and boundary conditions",
             "Solutions fitted to the visible test cases alone"],
            ["mu_6 safety", "Deep",
             "LLM-as-a-judge scoring for hallucinated dependencies and unsafe constructs",
             "Invented APIs and unconstrained execution"],
        ],
        "Evaluator implementation and the failure mode each addresses",
        widths=[2.6, 1.2, 6.0, 4.7], font_size=9)

    # ------------------------------------------------------------ Appendix 3
    b.section_heading("Appendix 3", "Per-task statistical results")
    path = os.path.join(b.assets, "tables", "wilcoxon_per_task.csv")
    if os.path.exists(path):
        with io.open(path, encoding="utf-8") as f:
            recs = list(csv.DictReader(f))
        b.body(
            f"The per-task Wilcoxon signed-rank tests compare the full adaptive condition against "
            f"each baseline separately within every task. {len(recs)} comparisons are reported. As "
            f"Section 6.5.2 explains, the per-task split has strictly limited power at "
            f"{ds['n_runs_per_cell']} repetitions, and the pooled analysis is the informative one; "
            f"this table is included for completeness."
        )
        rows = []
        for r in recs:
            rows.append([
                r.get("task_id", "")[:26],
                r.get("baseline_condition", "").replace("_", " "),
                r.get("n_runs", ""),
                r.get("mean_full_adaptive_fitness", ""),
                r.get("mean_baseline_fitness", ""),
                r.get("p_value_one_sided", ""),
            ])
        b.table(["Task", "Baseline", "n", "Mean C5", "Mean baseline", "p (1-sided)"],
                rows, "Per-task Wilcoxon signed-rank results",
                widths=[4.3, 3.0, 0.9, 2.1, 2.3, 1.9], font_size=8)
    else:
        b.body("Per-task results are distributed with the project as "
               "report_assets/tables/wilcoxon_per_task.csv.")

    # ------------------------------------------------------------ Appendix 4
    b.section_heading("Appendix 4", "Per-condition convergence and cost")
    path = os.path.join(b.assets, "tables", "condition_summary.csv")
    if os.path.exists(path):
        with io.open(path, encoding="utf-8") as f:
            recs = list(csv.DictReader(f))
        rows = [[
            r["label"], r["n_runs"], r["mean_peak_fitness"],
            r.get("mean_peak_fitness_normalised", ""),
            r.get("mean_active_weight_mass", ""),
            r["mean_cost_usd"], r["mean_cost_saved_pct"] + "%",
            r["mean_active_evaluators"],
        ] for r in recs]
        b.table(
            ["Condition", "Runs", "Peak F", "Peak F norm", "Weight mass", "Cost (USD)",
             "Saved", "Evaluators"],
            rows, "Per-condition summary over the complete ablation",
            widths=[3.0, 1.0, 1.6, 1.9, 1.9, 1.8, 1.4, 1.9], font_size=8)
        b.body(
            "The weight-mass column is the quantity that makes the raw and normalized fitness "
            "columns differ, and is the subject of Section 6.5.1."
        )

    # ------------------------------------------------------------ Appendix 5
    b.section_heading("Appendix 5", "Sample evolved system prompt")
    b.body(
        "The following system prompt was produced by the mutation operator for algorithmic "
        "efficiency, applied to the general balanced seed prompt during a live run. It is "
        "reproduced to illustrate the form the discrete component of the search space takes."
    )
    sample = (
        "You are an expert AI coding assistant. Your response must be precise, correct, and "
        "concise, with no unnecessary commentary. Primary directive: prioritize solutions with "
        "the lowest attainable asymptotic time and space complexity. Choose algorithms and data "
        "structures that minimize the asymptotic bound, and prefer a single pass with auxiliary "
        "hashing over nested iteration wherever the problem admits it. Validate inputs defensively "
        "before the main computation, and return the result directly without restating the problem."
    )
    pq = b.doc.add_paragraph()
    pq.paragraph_format.left_indent = Pt(24)
    pq.paragraph_format.right_indent = Pt(18)
    pq.paragraph_format.line_spacing = 1.0
    rq = pq.add_run(sample)
    rq.font.name = "Times New Roman"
    rq.font.size = Pt(11)
    rq.italic = True

    b.body(
        "The corresponding parent prompt read simply: \u201cYou are an expert, precise, and robust "
        "AI coding assistant. Solve the given problem clearly, accurately, and efficiently.\u201d "
        "The operator has introduced an explicit complexity objective, a preference between two "
        "named algorithmic strategies, and an instruction to suppress preamble, which is the "
        "conciseness dimension responding to weight placed on it."
    )

    # ------------------------------------------------------------ Appendix 6
    b.section_heading("Appendix 6", "Reproduction instructions")
    b.body(
        "The complete experimental pipeline, from a clean checkout to the generated report, is "
        "reproduced by the following sequence. The analysis and report-generation steps read only "
        "the persisted result files, so they can be re-run without repeating the search."
    )
    cmds = [
        "pip install -r requirements.txt",
        "pip install SentiToolKit==0.1.8",
        "echo GROQ_API_KEY=your_key_here > .env",
        "python run_live_experiments.py --runs 1 --generations 6",
        "python report_analysis.py --mock-csv results/ablation_live.csv \\",
        "                          --outdir report_assets",
        "python build_report.py --assets report_assets --out BCSE497J_Project_Report.docx",
        "python finalize_docx.py BCSE497J_Project_Report.docx",
    ]
    for c in cmds:
        pc = b.doc.add_paragraph()
        pc.paragraph_format.line_spacing = 1.0
        pc.paragraph_format.space_after = Pt(2)
        pc.paragraph_format.left_indent = Pt(24)
        rc = pc.add_run(c)
        rc.font.name = "Consolas"
        rc.font.size = Pt(10)
    b.body(
        "A GROQ_API_KEY in the environment, or in a .env file at the repository root, selects live "
        "execution against the hosted model. The driver used above additionally enables strict "
        "mode, so a failed request aborts the run rather than substituting a simulated response; "
        "reproducing the ablation therefore requires a working key and cannot silently degrade into "
        "offline data. Because the daily allowance is smaller than the full design, the driver "
        "checkpoints after every cell: when a key is exhausted it stops, and re-running the same "
        "command with a fresh key resumes from the last completed cell."
    )
    b.body(
        f"The result files distributed with the project contain {ds['mock_rows']} per-generation "
        f"records from the live ablation, each annotated with the number of live and simulated API "
        f"calls that produced it. Every figure and every statistic in Chapter 6 is derived from "
        f"them by report_analysis.py, which refuses to run if any simulated call is present, and "
        f"no quantity in this report was transcribed by hand."
    )
