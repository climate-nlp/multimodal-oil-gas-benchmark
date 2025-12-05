import React from 'react';
import { Link } from 'react-router-dom';

const BackIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="19" y1="12" x2="5" y2="12"></line>
        <polyline points="12 19 5 12 12 5"></polyline>
    </svg>
);

const ExternalLinkIcon = () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="inline ml-1">
        <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
        <polyline points="15 3 21 3 21 9"></polyline>
        <line x1="10" y1="14" x2="21" y2="3"></line>
    </svg>
);

export const References: React.FC = () => {
    return (
        <div className="min-h-screen bg-paper text-slate-900 font-sans">
            {/* Institutional Strip */}
            <div className="bg-slate-900 text-white border-b border-slate-800 py-2 px-4 sm:px-6 lg:px-8">
                <div className="max-w-7xl mx-auto flex justify-between items-center text-[10px] sm:text-xs font-medium tracking-widest uppercase">
                    <div className="flex items-center space-x-4 sm:space-x-6">
                        <div className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-princeton"></span>
                            <span>Princeton</span>
                        </div>
                        <div className="flex items-center gap-2">
                            <span className="w-1.5 h-1.5 rounded-full bg-stanford"></span>
                            <span>Stanford</span>
                        </div>
                        <div className="hidden sm:flex items-center gap-2 text-slate-400">
                            <span>Hitachi</span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Header */}
            <header className="bg-white border-b border-slate-200 py-6">
                <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
                    <Link
                        to="/"
                        className="inline-flex items-center gap-2 text-sm font-medium text-slate-600 hover:text-stanford transition-colors mb-4"
                    >
                        <BackIcon />
                        Back to Dashboard
                    </Link>
                    <h1 className="font-sans font-bold text-slate-900 text-3xl tracking-tight">
                        References & Disclaimers
                    </h1>
                    <p className="text-slate-600 mt-2">
                        Citations, methodological notes, and dataset information
                    </p>
                </div>
            </header>

            <main className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-12">

                {/* Disclaimer Section */}
                <section className="mb-12">
                    <div className="bg-amber-50 border-l-4 border-amber-500 p-6 rounded-sm mb-8">
                        <h2 className="font-sans font-bold text-amber-900 text-lg mb-3 flex items-center gap-2">
                            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
                                <line x1="12" y1="9" x2="12" y2="13"></line>
                                <line x1="12" y1="17" x2="12.01" y2="17"></line>
                            </svg>
                            Important Disclaimer
                        </h2>
                        <p className="text-amber-900 leading-relaxed">
                            This tool visualizes research findings from an academic benchmark dataset. The analysis represents
                            framing patterns in publicly available oil & gas advertising content and should not be interpreted
                            as definitive claims about individual companies or their intentions. All interpretations
                            are subject to the limitations of automated multimodal content analysis. This research evaluates
                            framing strategies employed in advertisements, not the truthfulness of claims or corporate environmental performance.
                        </p>
                    </div>
                </section>

                {/* Methodology Section */}
                <section className="mb-12">
                    <h2 className="font-sans font-bold text-slate-900 text-2xl mb-4 border-b border-slate-200 pb-2">
                        Methodology
                    </h2>
                    <div className="bg-white border border-slate-200 rounded-sm p-6 space-y-4">
                        <div>
                            <h3 className="font-semibold text-slate-900 mb-2">Data Collection</h3>
                            <p className="text-slate-700 leading-relaxed">
                                Video advertisements were collected from Facebook and YouTube, covering 706 videos (35,476 seconds
                                of footage) from more than 50 oil & gas companies and advocacy groups across 20 countries. The
                                Facebook domain includes 320 videos with climate obstruction framing labels, while the YouTube
                                domain contains 386 videos with expert-annotated impressionistic framing labels.
                            </p>
                        </div>
                        <div>
                            <h3 className="font-semibold text-slate-900 mb-2">Multimodal Frame Analysis</h3>
                            <p className="text-slate-700 leading-relaxed">
                                Our analysis employs vision-language models (VLMs) to identify narrative frames across visual
                                and textual content. The Facebook domain captures seven fine-grained climate obstruction frames
                                (CA, CB, GA, GC, PA, PB, SA). The YouTube domain identifies six impressionistic frames: Environment,
                                Green Innovation, Economy & Business, Work, Community & Life, and Patriotism.
                            </p>
                        </div>
                        <div>
                            <h3 className="font-semibold text-slate-900 mb-2">Limitations</h3>
                            <p className="text-slate-700 leading-relaxed">
                                Automated content analysis has inherent limitations. The dataset exhibits biases toward large
                                multinational corporations and English-language content. Facebook labels represent "distant annotations"
                                primarily based on ad text. YouTube annotations achieved a Fleiss' Kappa of 0.61, reflecting the
                                subjective nature of impressionistic framing. Model performance varies across labels, with some
                                categories (e.g., Green Innovation, Patriotism) proving particularly challenging.
                            </p>
                        </div>
                    </div>
                </section>

                {/* Key References */}
                <section className="mb-12">
                    <h2 className="font-sans font-bold text-slate-900 text-2xl mb-4 border-b border-slate-200 pb-2">
                        Key Academic References
                    </h2>
                    <div className="bg-slate-50 border border-slate-200 rounded-sm p-4 mb-4">
                        <p className="text-slate-700 text-sm">
                            <strong>Note:</strong> The references below represent key foundational works. For a complete list of references,
                            methodology details, and full citations, please see the <a href="https://github.com/climate-nlp/multimodal-oil-gas-benchmark" target="_blank" rel="noopener noreferrer" className="text-stanford hover:underline">full paper and supplementary materials</a>.
                        </p>
                    </div>
                    <div className="bg-white border border-slate-200 rounded-sm p-6">
                        <ol className="space-y-4 list-decimal list-inside">
                            <li className="text-slate-700 leading-relaxed pl-2">
                                <span className="font-medium">Holder, F., Mirza, S., Ngo-Lee, N., Carbone, J., & McKie, R. E.</span> (2023).
                                Climate obstruction and Facebook advertising: How a sample of climate obstruction organizations use social media to disseminate discourses of delay.
                                <em className="text-slate-600"> Climatic Change, 176</em>(2), 16.
                            </li>
                            <li className="text-slate-700 leading-relaxed pl-2">
                                <span className="font-medium">Rowlands, H., Morio, G., Tanner, D., & Manning, C.</span> (2024).
                                Predicting narratives of climate obstruction in social media advertising.
                                <em className="text-slate-600"> Findings of ACL 2024</em>, 5547-5558.
                            </li>
                            <li className="text-slate-700 leading-relaxed pl-2">
                                <span className="font-medium">Supran, G., & Oreskes, N.</span> (2021).
                                Rhetoric and frame analysis of ExxonMobil's climate change communications.
                                <em className="text-slate-600"> One Earth, 4</em>(5), 696-719.
                            </li>
                            <li className="text-slate-700 leading-relaxed pl-2">
                                <span className="font-medium">Entman, R. M.</span> (1993).
                                Framing: Toward clarification of a fractured paradigm.
                                <em className="text-slate-600"> Journal of Communication, 43</em>(4), 51-58.
                            </li>
                            <li className="text-slate-700 leading-relaxed pl-2">
                                <span className="font-medium">de Freitas Netto, S. V., Sobral, M. F. F., Ribeiro, A. R. B., & Soares, G. R. L.</span> (2020).
                                Concepts and forms of greenwashing: A systematic review.
                                <em className="text-slate-600"> Environmental Sciences Europe, 32</em>, 1-12.
                            </li>
                        </ol>
                    </div>
                </section>

                {/* Data Sources */}
                <section className="mb-12">
                    <h2 className="font-sans font-bold text-slate-900 text-2xl mb-4 border-b border-slate-200 pb-2">
                        Dataset & Code
                    </h2>
                    <div className="bg-white border border-slate-200 rounded-sm p-6">
                        <ul className="space-y-3">
                            <li className="flex items-start gap-2">
                                <span className="text-stanford mt-1">•</span>
                                <div>
                                    <a href="https://huggingface.co/datasets/climate-nlp/multimodal-oil-gas-benchmark" target="_blank" rel="noopener noreferrer" className="text-stanford hover:underline font-medium">
                                        HuggingFace Dataset
                                        <ExternalLinkIcon />
                                    </a>
                                    <p className="text-slate-600 text-sm mt-1">706 annotated video advertisements with framing labels and metadata</p>
                                </div>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="text-stanford mt-1">•</span>
                                <div>
                                    <a href="https://github.com/climate-nlp/multimodal-oil-gas-benchmark" target="_blank" rel="noopener noreferrer" className="text-stanford hover:underline font-medium">
                                        GitHub Repository
                                        <ExternalLinkIcon />
                                    </a>
                                    <p className="text-slate-600 text-sm mt-1">Code for benchmark experiments and evaluation pipeline</p>
                                </div>
                            </li>
                            <li className="flex items-start gap-2">
                                <span className="text-stanford mt-1">•</span>
                                <div>
                                    <span className="font-medium text-slate-900">Original Data Sources</span>
                                    <p className="text-slate-600 text-sm mt-1">Facebook Ads (via Meta Ad Library) and YouTube corporate channels</p>
                                </div>
                            </li>
                        </ul>
                    </div>
                </section>

                {/* Technical Stack */}
                <section className="mb-12">
                    <h2 className="font-sans font-bold text-slate-900 text-2xl mb-4 border-b border-slate-200 pb-2">
                        Technical Implementation
                    </h2>
                    <div className="bg-white border border-slate-200 rounded-sm p-6">
                        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                            <div>
                                <h3 className="font-semibold text-slate-900 mb-2">Dashboard</h3>
                                <ul className="text-slate-700 space-y-1 text-sm">
                                    <li>• React + TypeScript</li>
                                    <li>• Recharts for visualization</li>
                                    <li>• Tailwind CSS</li>
                                    <li>• Vite build system</li>
                                </ul>
                            </div>
                            <div>
                                <h3 className="font-semibold text-slate-900 mb-2">Models Benchmarked</h3>
                                <ul className="text-slate-700 space-y-1 text-sm">
                                    <li>• GPT-4.1 & GPT-4o-mini</li>
                                    <li>• Qwen2.5-VL (7B & 32B)</li>
                                    <li>• InternVL2 (8B)</li>
                                    <li>• DeepSeek-VL2 (4.5B)</li>
                                </ul>
                            </div>
                            <div>
                                <h3 className="font-semibold text-slate-900 mb-2">Processing</h3>
                                <ul className="text-slate-700 space-y-1 text-sm">
                                    <li>• PyTorch 2.6.0</li>
                                    <li>• HuggingFace Transformers</li>
                                    <li>• Whisper-1 (transcription)</li>
                                    <li>• CLIP embeddings</li>
                                </ul>
                            </div>
                            <div>
                                <h3 className="font-semibold text-slate-900 mb-2">Evaluation</h3>
                                <ul className="text-slate-700 space-y-1 text-sm">
                                    <li>• Multi-label classification</li>
                                    <li>• Zero-shot & 1-shot learning</li>
                                    <li>• F-score metrics</li>
                                    <li>• Cross-domain analysis</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </section>

                {/* Research Team */}
                <section className="mb-12">
                    <h2 className="font-sans font-bold text-slate-900 text-2xl mb-4 border-b border-slate-200 pb-2">
                        Research Team & Affiliations
                    </h2>
                    <div className="bg-white border border-slate-200 rounded-sm p-6">
                        <div className="mb-6 pb-4 border-b border-slate-200">
                            <p className="text-slate-700 leading-relaxed">
                                <span className="font-semibold">Gaku Morio</span><sup className="text-xs">1,2</sup>,
                                <span className="font-semibold"> Harri Rowlands</span><sup className="text-xs">3</sup>,
                                <span className="font-semibold"> Dominik Stammbach</span><sup className="text-xs">4</sup>,
                                <span className="font-semibold"> Christopher D. Manning</span><sup className="text-xs">2</sup>,
                                <span className="font-semibold"> Peter Henderson</span><sup className="text-xs">4</sup>
                            </p>
                        </div>
                        <div className="space-y-3">
                            <div className="flex items-start gap-3">
                                <span className="text-slate-500 text-sm font-mono">1</span>
                                <div>
                                    <p className="text-slate-700 text-sm">
                                        Hitachi, Ltd. & Hitachi America
                                    </p>
                                </div>
                            </div>
                            <div className="flex items-start gap-3">
                                <span className="text-slate-500 text-sm font-mono">2</span>
                                <div className="flex items-center gap-2">
                                    <p className="text-slate-700 text-sm font-medium">
                                        Stanford University
                                    </p>
                                </div>
                            </div>
                            <div className="flex items-start gap-3">
                                <span className="text-slate-500 text-sm font-mono">3</span>
                                <div>
                                    <p className="text-slate-700 text-sm">
                                        Centre for the Acceleration of Social Technology
                                    </p>
                                </div>
                            </div>
                            <div className="flex items-start gap-3">
                                <span className="text-slate-500 text-sm font-mono">4</span>
                                <div className="flex items-center gap-2">
                                    <p className="text-slate-700 text-sm font-medium">
                                        Princeton University
                                    </p>
                                </div>
                            </div>
                        </div>
                        <div className="mt-6 pt-4 border-t border-slate-200">
                            <p className="text-slate-600 text-sm">
                                For complete affiliation details, acknowledgments, and funding information, please see the full paper.
                            </p>
                        </div>
                    </div>
                </section>

                {/* License & Usage */}
                <section className="mb-12">
                    <h2 className="font-sans font-bold text-slate-900 text-2xl mb-4 border-b border-slate-200 pb-2">
                        License & Citation
                    </h2>
                    <div className="bg-slate-50 border border-slate-200 rounded-sm p-6">
                        <p className="text-slate-700 leading-relaxed mb-4">
                            This dataset is released under CC BY-NC 4.0 for academic and educational purposes.
                            Commercial use requires explicit permission from the research team. Individual video content
                            remains copyrighted by respective publishers.
                        </p>
                        <p className="text-slate-700 leading-relaxed mb-4">
                            When citing this work, please use:
                        </p>
                        <div className="bg-white border border-slate-300 rounded p-4 font-mono text-xs text-slate-800 mb-4">
                            @inproceedings{'{'}morio2024multimodal,<br />
                            &nbsp;&nbsp;title={'{'}A Multimodal Benchmark for Framing of Oil {'&'} Gas Advertising and Potential Greenwashing Detection{'}'},<br />
                            &nbsp;&nbsp;author={'{'}Morio, Gaku and Rowlands, Harri and Stammbach, Dominik and Manning, Christopher D. and Henderson, Peter{'}'},<br />
                            &nbsp;&nbsp;booktitle={'{'}Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track{'}'},<br />
                            &nbsp;&nbsp;year={'{'}2024{'}'}<br />
                            {'}'}
                        </div>
                        <p className="text-slate-600 text-sm">
                            For questions or collaboration inquiries, contact the authors via the email addresses provided in the paper.
                        </p>
                    </div>
                </section>

                {/* Footer */}
                <footer className="border-t border-slate-200 pt-8 mt-12">
                    <div className="text-center text-slate-500 text-sm">
                        <p>Dataset published at NeurIPS 2025 Datasets and Benchmarks Track</p>
                    </div>
                </footer>

            </main>
        </div>
    );
};

export default References;