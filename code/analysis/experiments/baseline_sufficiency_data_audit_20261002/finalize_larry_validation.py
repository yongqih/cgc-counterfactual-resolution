"""Save the audited operational proof and its scientific boundary."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.sparse import load_npz
from validate_larry_branches import exclude_ambiguous_late_records

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'larry'


def main():
    result = json.loads((ROOT / 'branch_validation_results.json').read_text(encoding='utf-8'))
    robust = json.loads((ROOT / 'branch_robustness_results.json').read_text(encoding='utf-8'))
    source = json.loads((ROOT / 'expression_source_manifest.json').read_text(encoding='utf-8'))
    genes = json.loads((ROOT / 'branch_validation_gene_names.json').read_text(encoding='utf-8'))
    meta = pd.read_csv(ROOT / 'branch_validation_metadata.csv')
    x = load_npz(ROOT / 'branch_validation_full_gene.npz')
    original = pd.read_csv(ROOT / 'stateFate_inVitro_metadata.txt.gz', sep='\t')
    assert x.shape == (len(meta), len(genes)) == (5036, 25289)
    assert np.isfinite(x.data).all() and (x.data >= 0).all()
    for name in original.columns:
        assert np.array_equal(meta[name].to_numpy(), original.iloc[meta.original_cell_index][name].to_numpy()), name
    clean, _, duplicate = exclude_ambiguous_late_records(meta, x)
    chosen = clean[clean.clone_index == 1978]
    assert len(chosen[chosen['Time point'] == 2]) == 2
    assert (chosen.loc[chosen['Time point'] == 2, 'Well'] == 0).all()
    end = chosen[chosen['Time point'] == 6]
    assert end.groupby('Well').size().to_dict() == {1: 17, 2: 33}
    assert duplicate['clone1978_records_excluded'] == 0
    significant = robust['all_207_clone_family_significant_results']
    assert len(significant) == 1 and significant[0]['clone_index'] == 1978
    assert significant[0]['p_holm_all_207_clones'] < .001
    assert result['clone1978_signal']['conditional_bootstrap_CI95'][0] > 0
    assert robust['reference_corrected_full_gene_sensitivity'][0]['conditional_bootstrap_CI95'][0] > 0
    assert robust['stronger_droplet_barcode_screen']['Neutrophil_p_Bonferroni_2277_tests'] < .001
    assert robust['stronger_droplet_barcode_screen']['full_gene_signal']['conditional_bootstrap_CI95'][0] > 0

    audit = {
        'status': 'Operational concept validated under explicit sampling and measurement assumptions',
        'claim': 'Measuring the full transcriptome of an available common antecedent does not guarantee identification of a unique realized future clonal branch transcriptome under a nominally matched protocol.',
        'input': 'All actual day-2 RNA records for a common clone, with all 25289 published gene features, before split culture; this includes two cells for clone 1978.',
        'output': 'Realized day-6 clonal population mean log1p RNA across all published genes, and cell composition, in each of two actual culture branches.',
        'logical_reason': 'Both branches share the same available pre-split record. If their true endpoint profiles differ, every deterministic function of that shared record has positive paired error. This does not depend on model training or choice.',
        'algebra': 'For any common vector prediction p: (||mu1-p||^2 + ||mu2-p||^2)/2 = ||p-(mu1+mu2)/2||^2 + ||mu1-mu2||^2/4.',
        'evidence': {'clone_index_zero_based': 1978,
                     'day6_well1': {'n': 17, 'Neutrophil': 6, 'Monocyte': 8, 'Undifferentiated': 3},
                     'day6_well2': {'n': 33, 'Neutrophil': 33, 'Monocyte': 0, 'Undifferentiated': 0},
                     'fate_test_family': 'All 207 common clones by all 11 published annotation classes, 2277 tests',
                     'Neutrophil_p_Holm': significant[0]['p_holm_all_207_clones'],
                     'full_gene_cross_library_contrast': result['clone1978_signal'],
                     'stronger_cell_identity_check': robust['stronger_droplet_barcode_screen'],
                     'duplicate_record_screen': duplicate},
        'noise_interpretation': 'Input measurement noise or sampling can remove response-relevant information and therefore qualify as an operational information limit. Independent zero-mean endpoint noise alone is not evidence of a true biological endpoint difference. Repeated cell samples and control contrasts support the endpoint contrast; systematic well-associated measurement bias remains an assumption.',
        'critical_assumptions': ['Clonal linkage is valid after known cross-family collisions are excluded.',
                                 'The sequenced cells adequately represent the two realized culture populations.',
                                 'The endpoint contrast is not entirely a systematic well-associated measurement artifact.',
                                 'For the cross-product interpretation, residual measurement/sampling errors have zero mean and are independent across disjoint library samples conditional on the realized populations.'],
        'not_established': result['not_established'] + ['A universally optimal numerical output resolution or a requirement to output fewer genes.',
                                                     'Two future daughter cells having exactly identical ideal biological starting states.',
                                                     'A decomposition of uncertainty into input measurement noise, missed starting-cell heterogeneity, future stochastic events and culture microenvironment.'],
        'inference_time_rule': 'Use the real shared early record and known protocol/time. Late cell annotations, control references, branch profiles and the paired midpoint are only validation or theoretical quantities, never proposed deployment inputs.',
        'publication_claim': 'The biological specificity of a predicted realized outcome must be justified by information available at prediction time; full-gene measurement alone does not supply that justification.',
        'primary_sources': ['https://pmc.ncbi.nlm.nih.gov/articles/PMC7608074/',
                            'https://doi.org/10.1126/science.aaw3381',
                            'https://github.com/AllonKleinLab/paper-data/tree/master/Lineage_tracing_on_transcriptional_landscapes_links_state_to_fate_during_differentiation',
                            'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140802'],
        'source_matrix': source,
        'verification': 'Retained expression dimensions, all cell-metadata alignments to original source row indices, finite nonnegative counts, pre-split well label, endpoint sample sizes, duplicate exclusion and final test outputs checked.',
        'retained_file_sha256': {}
    }
    for name in ['branch_validation_full_gene.npz', 'branch_validation_gene_names.json', 'branch_validation_metadata.csv']:
        audit['retained_file_sha256'][name] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (ROOT / 'validated_concept_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')

    report = f'''真实数据验证结论（2026-10-02）

核心结论
已经找到并验证一个直接的真实实例：完整的已测 baseline RNA，不能唯一决定后来某一个培养分支的实际 RNA 结局。该结论针对预测时实际拥有的共同早期记录，成立于下述明确的采样与测量假设；它不需要证明模型性能饱和，也不依赖某个模型拟合失败。

真实实验
Weinreb 等，Science 2020，LARRY，GSE140802。
研究者先给造血祖细胞加入谱系条码，day 2 取部分细胞测 RNA，随后将剩余细胞分到不同培养组，按相同名义分化方案培养，在 day 6 再测。作者的数据说明明确规定 day 2 的 Well=0，后续 Well=1 或 2。因此同一克隆的早期记录是两个后续培养组在分组前共同拥有的输入，不是把两个不同细胞的输入强行设为相等。

输入与输出
X：共同克隆的全部已测 day-2 RNA 记录，保留全部 25,289 个基因。主要实例有两个 early cells；两个培养组共享这份可用记录。
Y1、Y2：各培养组 day-6 该克隆的实际群体 RNA 轮廓；另用细胞组成解释 RNA 差异。
测到的 early cells 是被测姐妹细胞，不能冒充后来仍存活的每个 daughter cell 的直接测量。这是共享 baseline 对后续培养分支的检验。

结果
克隆编号 1978（公开 clone matrix 的零起始列号）：
组 1：17 个终点细胞，6 个中性粒细胞、8 个单核细胞、3 个未分化细胞。
组 2：33 个终点细胞，全部为中性粒细胞。
对全部 207 个可匹配克隆、全部 11 种注释类别共 2277 个检验统一做 Holm 校正后，中性粒细胞比例差异仍成立（校正 p={significant[0]['p_holm_all_207_clones']:.6g}）。
各组两个不同文库中的不重复细胞记录显示一致的组成与单核细胞 RNA 程序差异。单核细胞程序使用原论文 Table 2 的 10 个标记，未按当前克隆的表达差异选择基因。
更严格地排除整个原始元数据中同组、同日重复出现的任何 droplet barcode（包括不同克隆）后，主例子保留 17 对 32 个细胞，仍为 6 对 32 个中性粒细胞；按 2277 次检验作保守 Bonferroni 校正，p={robust['stronger_droplet_barcode_screen']['Neutrophil_p_Bonferroni_2277_tests']:.6g}，全基因差异交叉乘积区间仍为正。
全部 25,289 基因上的跨文库差异交叉乘积为正，其在已观察培养物内的细胞重采样区间也为正。按同文库其他克隆的中性粒细胞作参考校正后，方向保留。这是对文库偏差的敏感性检查，参考不被假设为严格生物学不变，不能作为独立生物学重复。
细胞类别与 RNA 程序来自同一 RNA 测量，不是两个独立实验。

数据问题与处理
42 个克隆编号跨独立培养来源出现，可能含谱系条码碰撞，全部排除；完整三方可匹配克隆从 217 降至 207。
同一时间、培养组、克隆和 droplet barcode 在不同文库中有可疑重复；主分析保守剔除每一份可疑记录，共 90 行、涉及 29 个克隆。主例子 1978 没有这种记录。
终点每组至少 10 个细胞的克隆有 21 个。细胞重采样不等于重新做培养实验；所有区间限于观察到的培养物。

为何这能越过“模型可能不够好”的质疑
令两个真实终点群体轮廓为 mu1、mu2。任何只依赖共同 X 的确定预测都必须给出同一个 p。
平均平方误差 = ||p-(mu1+mu2)/2||^2 + ||mu1-mu2||^2/4。
若真实 mu1 不等于 mu2，第二项严格为正。换模型、无限拟合能力或使用全部已测输入基因，都无法用同一个确定 p 同时等于两个实际终点。
mu1、mu2 和它们的中点是事后验证与推导中的量，不是预测时允许使用的输入。这里得到的是这对真实分支上的共同预测误差约束，没有估计跨所有生物场景的 Bayes 风险。

噪声是否也可以
可以：如果 baseline 的测量或采样噪声使影响后续响应的状态不可辨识，这已经是实际输入的信息限制。无需再证明理想无噪声 RNA 也不充分。
但数值可区分并不自动等于差异只有噪声；低准确率、非显著差异或图上重叠同样不够。终点仅有独立测量噪声造成的差异，也不足以证明真实生物响应不同。本例使用跨文库一致的细胞组成和多基因程序，并作检测能力和参考校正检查，以支持实际终点差异。
仍需假设：谱系连接正确、取样能代表所测培养物、差异不完全由培养组特异的系统测量偏差造成。隐藏培养环境若改变真实结局，是共同 baseline 未记录的信息；仪器偏差若只改变观测值，则是另一问题。

最强且成立的论文表述
“全基因 baseline 测量本身并不保证具体未来 RNA 结局的可辨识性；预测所声称的生物学细节必须由预测时可用的信息支持。”
不能扩大为“所有 baseline RNA 都无法预测全基因均值”，或“只能输出少数基因”。全基因的条件均值、分布和带不确定性的预测仍然合法。这里约束的是对具体结局的唯一指定和确定性，不是输出向量长度。
也不能把本例称为两个未来 daughter cells 的理想起始状态完全相同：实验仅证明它们的共同可用早期记录相同。生物随机性、未测姐妹细胞状态、输入测量噪声和微环境的贡献尚未分解。

证据文件与复现
图：larry/branch_validation.png
主结果：larry/branch_validation_results.json
全部克隆多重校正与参考检查：larry/branch_robustness_results.json
逐记录文库审计：larry/branch_library_cell_identity_audit.json
可疑重复排除：larry/branch_duplicate_record_screen.json
完整审计与文件哈希：larry/validated_concept_audit.json
保留真实表达子集：larry/branch_validation_full_gene.npz（5036 cells × 25289 genes；分析时再剔除可疑行）
原始来源总量归一化 UMI，不是整数 raw counts；分析使用 log1p，不称作 CP10k。
复现顺序：prepare_larry_branch_validation.py → audit_larry_library_identity.py → validate_larry_branches.py → audit_larry_branch_robustness.py → finalize_larry_validation.py。
已有保留子集时，从 validate_larry_branches.py 开始即可，不需要下载原始 2.07 GB 全矩阵。

原始来源
https://doi.org/10.1126/science.aaw3381
https://pmc.ncbi.nlm.nih.gov/articles/PMC7608074/
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140802
https://github.com/AllonKleinLab/paper-data/tree/master/Lineage_tracing_on_transcriptional_landscapes_links_state_to_fate_during_differentiation
'''
    (HERE / '真实数据验证结论.txt').write_text(report, encoding='utf-8')
    findings = json.loads((HERE / 'findings.json').read_text(encoding='utf-8'))
    findings['conclusion'] = audit['status'] + ': ' + audit['claim'] + ' The unrestricted noiseless single-daughter-RNA claim and impossibility of full-gene conditional-mean prediction are not established.'
    findings['completed_validation'] = audit
    for candidate in findings.get('candidates', []):
        if 'LARRY' in candidate.get('name', ''):
            candidate['operational_validation'] = 'Direct shared-antecedent split-culture test supports non-identifiability of realized day-6 clonal RNA outcomes; see completed_validation and validated_concept_audit.json. Existing MI objections remain valid and are not used for this proof.'
    (HERE / 'findings.json').write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': audit['status'], 'input_output_alignment': 'verified',
                      'report': str(HERE / '真实数据验证结论.txt'), 'p_Holm': significant[0]['p_holm_all_207_clones']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
