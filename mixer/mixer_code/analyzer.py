import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
import torch.nn.functional as F

class RepresentationAnalyzer:
    """
    Analyze neural network representations using various metrics including:
    - Centered Kernel Alignment (CKA)
    - Silhouette Score
    - Isotropy measures
    - Neural Collapse metrics
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
    
    def center_gram_matrix(self, gram):
        """Center a Gram matrix"""
        n = gram.shape[0]
        unit = torch.ones(n, n, device=gram.device) / n
        gram_centered = gram - unit @ gram - gram @ unit + unit @ gram @ unit
        return gram_centered
    
    def cka(self, X, Y):
        """
        Compute Centered Kernel Alignment (CKA) between two feature matrices
        
        Args:
            X: Features from model 1, shape (n_samples, n_features)
            Y: Features from model 2, shape (n_samples, n_features)
            
        Returns:
            CKA score (float)
            
        Reference: "Similarity of Neural Network Representations Revisited" 
                  (Kornblith et al., 2019)
        """
        # Ensure tensors are on the same device
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Compute Gram matrices
        gram_X = X @ X.T
        gram_Y = Y @ Y.T
        
        # Center the Gram matrices
        gram_X_centered = self.center_gram_matrix(gram_X)
        gram_Y_centered = self.center_gram_matrix(gram_Y)
        
        # Compute CKA
        numerator = torch.trace(gram_X_centered @ gram_Y_centered)
        denominator = torch.sqrt(
            torch.trace(gram_X_centered @ gram_X_centered) * 
            torch.trace(gram_Y_centered @ gram_Y_centered)
        )
        
        return (numerator / denominator).item()
    
    def linear_cka(self, X, Y):
        """
        Compute Linear CKA (faster version for large matrices)
        
        Args:
            X: Features from model 1, shape (n_samples, n_features)
            Y: Features from model 2, shape (n_samples, n_features)
            
        Returns:
            Linear CKA score (float)
        """
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # Center the features
        X_centered = X - X.mean(dim=0, keepdim=True)
        Y_centered = Y - Y.mean(dim=0, keepdim=True)
        
        # Compute cross-covariance matrix
        cov_XY = X_centered.T @ Y_centered
        
        # Compute CKA
        numerator = torch.trace(cov_XY @ cov_XY.T)
        denominator = torch.sqrt(
            torch.trace(X_centered.T @ X_centered @ X_centered.T @ X_centered) *
            torch.trace(Y_centered.T @ Y_centered @ Y_centered.T @ Y_centered)
        )
        
        return (numerator / denominator).item()
    
    def silhouette_analysis(self, features, labels, n_clusters=None):
        """
        Compute silhouette score for feature representations
        
        Args:
            features: Feature matrix, shape (n_samples, n_features)
            labels: True labels for samples
            n_clusters: Number of clusters for K-means (if None, uses true labels)
            
        Returns:
            dict with silhouette scores and cluster assignments
            
        Reference: Used extensively in clustering evaluation literature
        """
        features_np = features.cpu().numpy() if torch.is_tensor(features) else features
        labels_np = labels.cpu().numpy() if torch.is_tensor(labels) else labels
        
        results = {}
        
        # Silhouette score with true labels
        true_silhouette = silhouette_score(features_np, labels_np)
        results['true_labels_silhouette'] = true_silhouette
        
        # Silhouette score with K-means clustering
        if n_clusters is None:
            n_clusters = len(np.unique(labels_np))
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_np)
        kmeans_silhouette = silhouette_score(features_np, cluster_labels)
        
        results['kmeans_silhouette'] = kmeans_silhouette
        results['cluster_labels'] = cluster_labels
        results['n_clusters'] = n_clusters
        
        return results
    
    def isotropy_measure(self, features):
        """
        Measure isotropy of feature representations
        
        Args:
            features: Feature matrix, shape (n_samples, n_features)
            
        Returns:
            dict with isotropy metrics
            
        Reference: "On the Isotropy of Word Embeddings" and related works
        """
        features = features.to(self.device)
        
        # Center the features
        features_centered = features - features.mean(dim=0, keepdim=True)
        
        # Compute covariance matrix
        cov_matrix = torch.cov(features_centered.T)
        
        # Compute eigenvalues
        eigenvals = torch.linalg.eigvals(cov_matrix).real
        eigenvals = torch.sort(eigenvals, descending=True)[0]
        
        # Isotropy measures
        # 1. Condition number (ratio of largest to smallest eigenvalue)
        condition_number = (eigenvals[0] / eigenvals[-1]).item()
        
        # 2. Effective rank (related to isotropy)
        eigenvals_normalized = eigenvals / eigenvals.sum()
        entropy = -torch.sum(eigenvals_normalized * torch.log(eigenvals_normalized + 1e-12))
        effective_rank = torch.exp(entropy).item()
        
        # 3. Isotropy coefficient (1 - variance of normalized eigenvalues)
        eigenvals_norm = eigenvals / eigenvals.sum()
        isotropy_coeff = 1 - torch.var(eigenvals_norm).item()
        
        return {
            'condition_number': condition_number,
            'effective_rank': effective_rank,
            'isotropy_coefficient': isotropy_coeff,
            'eigenvalues': eigenvals.cpu().numpy()
        }
    
    def neural_collapse_metrics(self, features, labels):
        """
        Compute Neural Collapse related metrics
        
        Args:
            features: Feature matrix, shape (n_samples, n_features)
            labels: Class labels
            
        Returns:
            dict with neural collapse metrics
            
        Reference: "Prevalence of Neural Collapse during the terminal phase of deep learning training"
                  (Papyan et al., 2020)
        """
        features = features.to(self.device)
        labels = labels.to(self.device)
        
        unique_labels = torch.unique(labels)
        num_classes = len(unique_labels)
        
        # Compute class means (centroids)
        class_means = []
        for label in unique_labels:
            mask = labels == label
            class_mean = features[mask].mean(dim=0)
            class_means.append(class_mean)
        
        class_means = torch.stack(class_means)  # (num_classes, feature_dim)
        
        # Global mean
        global_mean = features.mean(dim=0)
        
        # NC1: Variability collapse - within-class variability
        within_class_variance = 0
        total_samples = 0
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_features = features[mask]
            n_samples = class_features.shape[0]
            
            if n_samples > 1:
                # Compute within-class covariance
                centered_features = class_features - class_means[i]
                class_var = torch.trace(torch.cov(centered_features.T))
                within_class_variance += class_var * n_samples
                total_samples += n_samples
        
        within_class_variance /= total_samples
        
        # NC2: Convergence to simplex ETF (Equiangular Tight Frame)
        # Center class means
        centered_class_means = class_means - global_mean
        
        # Compute pairwise cosine similarities between class means
        normalized_means = F.normalize(centered_class_means, dim=1)
        cosine_similarities = normalized_means @ normalized_means.T
        
        # Remove diagonal (self-similarities)
        mask = ~torch.eye(num_classes, dtype=bool, device=self.device)
        off_diagonal_cosines = cosine_similarities[mask]
        
        # For a perfect simplex ETF, all off-diagonal cosines should be -1/(C-1)
        ideal_cosine = -1.0 / (num_classes - 1) if num_classes > 1 else 0
        etf_deviation = torch.std(off_diagonal_cosines - ideal_cosine).item()
        
        # NC3: Convergence of classifier weights to class means
        # (This would require access to the classifier layer)
        
        # NC4: Alignment between features and class means
        # For each sample, compute cosine similarity with its class mean vs other class means
        correct_alignments = []
        incorrect_alignments = []
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_features = features[mask]
            
            if class_features.shape[0] > 0:
                # Normalize features and class means
                norm_features = F.normalize(class_features, dim=1)
                norm_class_means = F.normalize(centered_class_means, dim=1)
                
                # Cosine similarity with correct class mean
                correct_sim = (norm_features * norm_class_means[i]).sum(dim=1)
                correct_alignments.extend(correct_sim.cpu().tolist())
                
                # Average cosine similarity with incorrect class means
                incorrect_means = torch.cat([norm_class_means[:i], norm_class_means[i+1:]])
                if len(incorrect_means) > 0:
                    incorrect_sims = norm_features @ incorrect_means.T
                    avg_incorrect_sim = incorrect_sims.mean(dim=1)
                    incorrect_alignments.extend(avg_incorrect_sim.cpu().tolist())
        
        # Compute alignment ratio
        if incorrect_alignments:
            alignment_ratio = np.mean(correct_alignments) / np.mean(np.abs(incorrect_alignments))
        else:
            alignment_ratio = float('inf')
        
        return {
            'within_class_variance': within_class_variance.item(),
            'etf_deviation': etf_deviation,
            'mean_correct_alignment': np.mean(correct_alignments),
            'mean_incorrect_alignment': np.mean(incorrect_alignments) if incorrect_alignments else 0,
            'alignment_ratio': alignment_ratio,
            'class_means_cosine_std': torch.std(off_diagonal_cosines).item()
        }

def extract_features_hook(model, layer_name, input_data):
    """
    Helper function to extract features from a specific layer using hooks
    
    Args:
        model: PyTorch model
        layer_name: Name of the layer to extract features from
        input_data: Input data batch
        
    Returns:
        Extracted features
    """
    features = []
    
    def hook_fn(module, input, output):
        # Flatten spatial dimensions if needed (for conv layers)
        if len(output.shape) > 2:
            output = output.view(output.size(0), -1)
        features.append(output.detach())
    
    # Get the layer by name
    layer = dict(model.named_modules())[layer_name]
    handle = layer.register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_data)
    
    # Remove hook
    handle.remove()
    
    return torch.cat(features, dim=0) if features else None

def print_comparison_results(cka_score, isotropy_acn, isotropy_resnet, nc_acn, nc_resnet, layer_name=""):
    """
    Print formatted comparison results between ACN and ResNet representations
    """
    print(f"\n{'='*60}")
    print(f"REPRESENTATION ANALYSIS{' - ' + layer_name if layer_name else ''}")
    print(f"{'='*60}")
    
    # CKA Score
    print(f"\n📊 SIMILARITY BETWEEN ARCHITECTURES:")
    print(f"   CKA Score: {cka_score:.4f}")
    if cka_score > 0.8:
        print(f"   → Very similar representations")
    elif cka_score > 0.5:
        print(f"   → Moderately similar representations")
    else:
        print(f"   → Different representations")
    
    # Isotropy Comparison
    print(f"\n🎯 ISOTROPY ANALYSIS:")
    print(f"   ACN  - Condition Number: {isotropy_acn['condition_number']:.2f}")
    print(f"   ResNet - Condition Number: {isotropy_resnet['condition_number']:.2f}")
    print(f"   ACN  - Effective Rank: {isotropy_acn['effective_rank']:.2f}")
    print(f"   ResNet - Effective Rank: {isotropy_resnet['effective_rank']:.2f}")
    print(f"   ACN  - Isotropy Coeff: {isotropy_acn['isotropy_coefficient']:.4f}")
    print(f"   ResNet - Isotropy Coeff: {isotropy_resnet['isotropy_coefficient']:.4f}")
    
    # Determine which is more isotropic
    if isotropy_acn['isotropy_coefficient'] > isotropy_resnet['isotropy_coefficient']:
        print(f"   → ACN features are more isotropic (better spread)")
    else:
        print(f"   → ResNet features are more isotropic (better spread)")
    
    # Neural Collapse Comparison
    print(f"\n🔄 NEURAL COLLAPSE ANALYSIS:")
    print(f"   ACN  - Within-class Variance: {nc_acn['within_class_variance']:.4f}")
    print(f"   ResNet - Within-class Variance: {nc_resnet['within_class_variance']:.4f}")
    print(f"   ACN  - ETF Deviation: {nc_acn['etf_deviation']:.4f}")
    print(f"   ResNet - ETF Deviation: {nc_resnet['etf_deviation']:.4f}")
    print(f"   ACN  - Alignment Ratio: {nc_acn['alignment_ratio']:.2f}")
    print(f"   ResNet - Alignment Ratio: {nc_resnet['alignment_ratio']:.2f}")
    
    # Neural collapse interpretation
    if nc_acn['within_class_variance'] < nc_resnet['within_class_variance']:
        print(f"   → ACN shows stronger variability collapse (NC1)")
    else:
        print(f"   → ResNet shows stronger variability collapse (NC1)")
    
    if nc_acn['etf_deviation'] < nc_resnet['etf_deviation']:
        print(f"   → ACN class means closer to simplex structure (NC2)")
    else:
        print(f"   → ResNet class means closer to simplex structure (NC2)")
    
    if nc_acn['alignment_ratio'] > nc_resnet['alignment_ratio']:
        print(f"   → ACN features better aligned with class structure (NC4)")
    else:
        print(f"   → ResNet features better aligned with class structure (NC4)")
    
    print(f"\n{'='*60}")

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = RepresentationAnalyzer()
    
    # Example with random data
    torch.manual_seed(42)
    features_model1 = torch.randn(1000, 512)  # 1000 samples, 512-dim features
    features_model2 = torch.randn(1000, 512)  # Features from another model
    labels = torch.randint(0, 10, (1000,))    # 10 classes
    
    # Compute CKA
    cka_score = analyzer.cka(features_model1, features_model2)
    print(f"CKA Score: {cka_score:.4f}")
    
    # Compute Linear CKA (faster)
    linear_cka_score = analyzer.linear_cka(features_model1, features_model2)
    print(f"Linear CKA Score: {linear_cka_score:.4f}")
    
    # Silhouette analysis
    silhouette_results = analyzer.silhouette_analysis(features_model1, labels)
    print(f"True Labels Silhouette: {silhouette_results['true_labels_silhouette']:.4f}")
    print(f"K-means Silhouette: {silhouette_results['kmeans_silhouette']:.4f}")
    
    # Isotropy measures
    isotropy_results = analyzer.isotropy_measure(features_model1)
    print(f"Condition Number: {isotropy_results['condition_number']:.2f}")
    print(f"Effective Rank: {isotropy_results['effective_rank']:.2f}")
    print(f"Isotropy Coefficient: {isotropy_results['isotropy_coefficient']:.4f}")
    
    # Neural collapse metrics
    nc_results = analyzer.neural_collapse_metrics(features_model1, labels)
    print(f"Within-class Variance: {nc_results['within_class_variance']:.4f}")
    print(f"ETF Deviation: {nc_results['etf_deviation']:.4f}")
    print(f"Alignment Ratio: {nc_results['alignment_ratio']:.2f}")
