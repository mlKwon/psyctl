"""Multi-metric consensus layer analyzer."""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression  # type: ignore[import-not-found]
from sklearn.model_selection import cross_val_score  # type: ignore[import-not-found]
from sklearn.svm import LinearSVC  # type: ignore[import-not-found]

from psyctl.core.analyzers.base import BaseLayerAnalyzer
from psyctl.core.logger import get_logger


class ConsensusAnalyzer(BaseLayerAnalyzer):
    """
    Multi-metric consensus analysis for layer quality assessment.

    Combines multiple metrics to robustly evaluate how well a layer
    can separate positive and neutral activations:

    1. **Linear SVM accuracy**: Basic separability (current baseline)
    2. **Logistic Regression accuracy**: L2-regularized linear classifier
    3. **Fisher Discriminant Ratio**: Measures class separability without overfitting
       FDR = (μ₁ - μ₂)² / (σ₁² + σ₂²)
    4. **Effective Rank**: Detects representational collapse
       ER = (Σs)² / Σ(s²) where s are singular values
    5. **Cross-validated accuracy**: SVM with 5-fold CV to prevent overfitting

    The consensus score is a weighted average of normalized metrics,
    providing a robust assessment that no single metric can game.

    Advantages over SVM-only:
    - Prevents overfitting (cross-validation)
    - Detects representational collapse (effective rank)
    - More robust (consensus of multiple metrics)
    - No hyperparameter tuning needed (fixed weights)

    Attributes:
        logger: Logger instance
        metric_weights: Weights for each metric in consensus
    """

    def __init__(
        self,
        svm_weight: float = 0.15,
        logreg_weight: float = 0.15,
        fisher_weight: float = 0.25,
        rank_weight: float = 0.15,
        cv_weight: float = 0.30,
    ):
        """
        Initialize ConsensusAnalyzer.

        Args:
            svm_weight: Weight for SVM accuracy (default: 0.15)
            logreg_weight: Weight for LogReg accuracy (default: 0.15)
            fisher_weight: Weight for Fisher ratio (default: 0.25)
            rank_weight: Weight for effective rank (default: 0.15)
            cv_weight: Weight for cross-validated accuracy (default: 0.30)

        Note: Weights should sum to 1.0. CV gets highest weight as it's
              the most reliable metric (prevents overfitting).
        """
        self.logger = get_logger("consensus_analyzer")

        # Validate weights
        total_weight = (
            svm_weight + logreg_weight + fisher_weight + rank_weight + cv_weight
        )
        if not np.isclose(total_weight, 1.0):
            self.logger.warning(
                f"Metric weights sum to {total_weight:.3f}, not 1.0. "
                "Normalizing weights."
            )
            norm = total_weight
            svm_weight /= norm
            logreg_weight /= norm
            fisher_weight /= norm
            rank_weight /= norm
            cv_weight /= norm

        self.metric_weights = {
            "svm_acc": svm_weight,
            "logreg_acc": logreg_weight,
            "fisher": fisher_weight,
            "rank": rank_weight,
            "cv_acc": cv_weight,
        }

        self.logger.debug(f"Initialized with metric weights: {self.metric_weights}")

    def analyze(
        self,
        positive_activations: list[torch.Tensor],
        neutral_activations: list[torch.Tensor],
    ) -> dict[str, float]:
        """
        Analyze separation using multi-metric consensus.

        Args:
            positive_activations: List of activation tensors
            neutral_activations: List of activation tensors

        Returns:
            Dictionary with metrics:
            - score: Consensus metric (weighted average)
            - svm_acc: Linear SVM training accuracy
            - logreg_acc: Logistic Regression training accuracy
            - fisher: Fisher Discriminant Ratio
            - rank: Effective rank (normalized)
            - cv_acc: 5-fold cross-validated SVM accuracy
            - converged: Whether all classifiers converged
        """
        self.logger.debug(
            f"Analyzing {len(positive_activations)} positive and "
            f"{len(neutral_activations)} neutral activations"
        )

        # Convert to numpy arrays
        x_pos = self._stack_tensors(positive_activations)
        x_neu = self._stack_tensors(neutral_activations)

        # Combine data and create labels
        x = np.vstack([x_pos, x_neu])  # [N, D]
        y = np.hstack([np.zeros(len(x_pos)), np.ones(len(x_neu))])  # [N]

        n_samples, n_features = x.shape

        self.logger.debug(f"Dataset: {n_samples} samples x {n_features} features")

        # Ensure we have enough samples for cross-validation
        if n_samples < 10:
            self.logger.warning(
                f"Only {n_samples} samples available. "
                "Consensus analysis may be unreliable."
            )

        metrics = {}
        converged = True

        # 1. Linear SVM accuracy
        try:
            svm = LinearSVC(C=1.0, max_iter=10000, dual="auto", random_state=42)
            svm.fit(x, y)
            metrics["svm_acc"] = float(svm.score(x, y))
            self.logger.debug(f"SVM accuracy: {metrics['svm_acc']:.4f}")
        except Exception as e:
            self.logger.error(f"SVM training failed: {e}")
            metrics["svm_acc"] = 0.0
            converged = False

        # 2. Logistic Regression accuracy (L2 regularized)
        try:
            logreg = LogisticRegression(
                penalty="l2",
                C=1.0,
                max_iter=10000,
                random_state=42,
                solver="lbfgs",
            )
            logreg.fit(x, y)
            metrics["logreg_acc"] = float(logreg.score(x, y))
            self.logger.debug(f"LogReg accuracy: {metrics['logreg_acc']:.4f}")
        except Exception as e:
            self.logger.error(f"LogReg training failed: {e}")
            metrics["logreg_acc"] = 0.0
            converged = False

        # 3. Fisher Discriminant Ratio
        try:
            fisher_ratio = self._compute_fisher_ratio(x_pos, x_neu)
            metrics["fisher"] = float(fisher_ratio)
            self.logger.debug(f"Fisher ratio: {metrics['fisher']:.4f}")
        except Exception as e:
            self.logger.error(f"Fisher ratio computation failed: {e}")
            metrics["fisher"] = 0.0

        # 4. Effective Rank (normalized)
        try:
            effective_rank = self._compute_effective_rank(x)
            # Normalize by dividing by min dimension
            normalized_rank = effective_rank / min(n_samples, n_features)
            metrics["rank"] = float(normalized_rank)
            self.logger.debug(
                f"Effective rank: {effective_rank:.2f} "
                f"(normalized: {metrics['rank']:.4f})"
            )
        except Exception as e:
            self.logger.error(f"Effective rank computation failed: {e}")
            metrics["rank"] = 0.0

        # 5. Cross-validated accuracy (most important metric)
        try:
            if n_samples >= 10:  # Need at least 10 samples for 5-fold CV
                svm_cv = LinearSVC(C=1.0, max_iter=10000, dual="auto", random_state=42)
                cv_scores = cross_val_score(svm_cv, x, y, cv=min(5, n_samples // 2))
                metrics["cv_acc"] = float(cv_scores.mean())
                self.logger.debug(
                    f"CV accuracy: {metrics['cv_acc']:.4f} (±{cv_scores.std():.4f})"
                )
            else:
                # Not enough samples for CV, fall back to train accuracy
                self.logger.warning(
                    f"Only {n_samples} samples, using SVM train acc for CV metric"
                )
                metrics["cv_acc"] = metrics["svm_acc"]
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            metrics["cv_acc"] = 0.0
            converged = False

        # Compute consensus score
        consensus_score = sum(
            self.metric_weights[key] * metrics[key] for key in self.metric_weights
        )

        metrics["score"] = float(consensus_score)
        metrics["converged"] = converged

        self.logger.info(
            f"Consensus analysis: score={metrics['score']:.4f}, "
            f"svm={metrics['svm_acc']:.3f}, "
            f"cv={metrics['cv_acc']:.3f}, "
            f"fisher={metrics['fisher']:.3f}, "
            f"rank={metrics['rank']:.3f}"
        )

        return metrics

    def get_metric_contributions(self, metrics: dict[str, float]) -> dict[str, float]:
        """
        Calculate each metric's weighted contribution to the consensus score.

        Args:
            metrics: Dictionary of metric values from analyze()

        Returns:
            Dictionary mapping metric names to their weighted contributions

        Example:
            >>> metrics = analyzer.analyze(pos_acts, neu_acts)
            >>> contributions = analyzer.get_metric_contributions(metrics)
            >>> # contributions = {"svm_acc": 0.12, "cv_acc": 0.27, ...}
        """
        contributions = {}
        for metric_name, weight in self.metric_weights.items():
            metric_value = metrics.get(metric_name, 0.0)
            contributions[metric_name] = weight * metric_value

        return contributions

    def get_metric_agreement(self, metrics: dict[str, float]) -> float:
        """
        Calculate metric agreement score (inverse of standard deviation).

        High agreement (low std) indicates all metrics agree on layer quality.
        Low agreement (high std) indicates metrics disagree, suggesting caution.

        Args:
            metrics: Dictionary of metric values from analyze()

        Returns:
            Agreement score in [0, 1] range:
            - 1.0 = perfect agreement (all metrics identical)
            - 0.5 = moderate disagreement
            - 0.0 = extreme disagreement

        Example:
            >>> metrics = analyzer.analyze(pos_acts, neu_acts)
            >>> agreement = analyzer.get_metric_agreement(metrics)
            >>> if agreement < 0.5:
            ...     print("Warning: Metrics disagree substantially")
        """
        # Extract metric values (excluding score and converged)
        metric_values = [metrics.get(key, 0.0) for key in self.metric_weights]

        # Calculate standard deviation
        std = np.std(metric_values)

        # Convert std to agreement score [0, 1]
        # std = 0 -> agreement = 1.0 (perfect agreement)
        # std = 0.5 -> agreement = 0.0 (extreme disagreement)
        # Using sigmoid-like transformation
        agreement = 1.0 / (1.0 + 2.0 * std)

        return float(agreement)

    def explain_score(self, metrics: dict[str, float]) -> str:
        """
        Generate human-readable explanation of the consensus score.

        Args:
            metrics: Dictionary of metric values from analyze()

        Returns:
            Multi-line string explaining the consensus score

        Example:
            >>> metrics = analyzer.analyze(pos_acts, neu_acts)
            >>> print(analyzer.explain_score(metrics))
            Consensus Score: 0.85 (Excellent)

            Metric Contributions:
              Cross-validation: 0.27 (30% weight, value: 0.90)
              Fisher ratio: 0.21 (25% weight, value: 0.85)
              ...

            Overall Assessment: Strong separation with high confidence
        """
        score = metrics.get("score", 0.0)
        contributions = self.get_metric_contributions(metrics)
        agreement = self.get_metric_agreement(metrics)

        # Classify score
        if score >= 0.9:
            quality = "Excellent"
            assessment = "Outstanding separation with very high confidence"
        elif score >= 0.8:
            quality = "Very Good"
            assessment = "Strong separation with high confidence"
        elif score >= 0.7:
            quality = "Good"
            assessment = "Good separation, reliable for steering"
        elif score >= 0.6:
            quality = "Moderate"
            assessment = "Moderate separation, usable but with caution"
        elif score >= 0.5:
            quality = "Fair"
            assessment = "Weak separation, may not be reliable"
        else:
            quality = "Poor"
            assessment = "Very weak or no separation, not recommended"

        # Build explanation
        lines = [
            f"Consensus Score: {score:.3f} ({quality})",
            "",
            "Metric Contributions:",
        ]

        # Sort contributions by value (descending)
        sorted_contribs = sorted(
            contributions.items(), key=lambda x: x[1], reverse=True
        )

        metric_names_display = {
            "cv_acc": "Cross-validation",
            "fisher": "Fisher ratio",
            "svm_acc": "SVM accuracy",
            "logreg_acc": "LogReg accuracy",
            "rank": "Effective rank",
        }

        for metric_name, contribution in sorted_contribs:
            display_name = metric_names_display.get(metric_name, metric_name)
            weight = self.metric_weights[metric_name]
            value = metrics.get(metric_name, 0.0)
            lines.append(
                f"  {display_name}: {contribution:.3f} "
                f"({weight * 100:.0f}% weight, value: {value:.3f})"
            )

        lines.append("")
        lines.append(f"Metric Agreement: {agreement:.3f}")

        if agreement < 0.5:
            lines.append(
                "  Warning: Metrics disagree substantially. "
                "Interpret consensus with caution."
            )
        elif agreement >= 0.8:
            lines.append("  All metrics strongly agree on layer quality.")

        lines.append("")
        lines.append(f"Overall Assessment: {assessment}")

        if not metrics.get("converged", True):
            lines.append("")
            lines.append(
                "  Warning: Some classifiers did not converge. "
                "Results may be unreliable."
            )

        return "\n".join(lines)

    def _compute_fisher_ratio(self, x_pos: np.ndarray, x_neu: np.ndarray) -> float:
        """
        Compute normalized Fisher Discriminant Ratio.

        FDR = (μ₁ - μ₂)² / (σ₁² + σ₂²)

        Averaged over all features, then normalized to [0, 1] range using
        sigmoid function to ensure fair comparison with other metrics.

        Args:
            x_pos: Positive activations [N_pos, D]
            x_neu: Neutral activations [N_neu, D]

        Returns:
            Normalized Fisher ratio in [0, 1] range
            - 0.5 indicates no separation
            - >0.5 indicates good separation
            - <0.5 indicates negative separation (unlikely with proper labels)
        """
        # Compute means
        mu_pos = x_pos.mean(axis=0)  # [D]
        mu_neu = x_neu.mean(axis=0)  # [D]

        # Compute variances
        var_pos = x_pos.var(axis=0)  # [D]
        var_neu = x_neu.var(axis=0)  # [D]

        # Compute Fisher ratio per feature
        # Add small epsilon to avoid division by zero
        numerator = (mu_pos - mu_neu) ** 2
        denominator = var_pos + var_neu + 1e-8
        fisher_per_feature = numerator / denominator

        # Get mean Fisher ratio (unbounded, [0, ∞))
        raw_fisher = fisher_per_feature.mean()

        # Normalize to [0, 1] using sigmoid function
        # This ensures fair weighting with other bounded metrics
        normalized_fisher = 1.0 / (1.0 + np.exp(-raw_fisher))

        return normalized_fisher

    def _compute_effective_rank(self, x: np.ndarray) -> float:
        """
        Compute effective rank using participation ratio.

        ER = (Σs)² / Σ(s²)

        where s are singular values. This measures how many
        "effective dimensions" the data spans.

        Low rank → representational collapse
        High rank → rich, diverse representations

        Args:
            x: Activation matrix [N, D]

        Returns:
            Effective rank (between 1 and min(N, D))
        """
        # Compute SVD (only singular values needed)
        _, singular_values, _ = np.linalg.svd(x, full_matrices=False)

        # Compute participation ratio
        sum_s = singular_values.sum()
        sum_s_squared = (singular_values**2).sum()

        # Handle edge cases
        if sum_s_squared < 1e-10:
            return 1.0  # Degenerate case

        effective_rank = (sum_s**2) / sum_s_squared

        return effective_rank
