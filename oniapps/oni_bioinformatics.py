import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import logging
import io
import base64
from datetime import datetime
import json
import time
import re
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# App metadata
APP_NAME = "oni_bioinformatics"
APP_DESCRIPTION = "Bioinformatics analysis tools for ONI"
APP_VERSION = "1.0.0"
APP_AUTHOR = "ONI Team"
APP_CATEGORY = "Science"
APP_DEPENDENCIES = ["numpy", "matplotlib", "biopython"]
APP_DEFAULT = False

class ONIBioinformatics:
    """
    ONI Bioinformatics - Analysis tools for biological data.
    
    Provides capabilities for:
    - DNA/RNA sequence analysis
    - Protein sequence analysis
    - Sequence alignment
    - Phylogenetic analysis
    - Structural bioinformatics
    - Genomic data visualization
    """
    
    def __init__(self):
        """Initialize the ONI Bioinformatics module."""
        self.sequences = {}
        self.alignments = {}
        self.analysis_results = {}
        self.figure_counter = 0
        
        # Genetic code (standard)
        self.genetic_code = {
            'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
            'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
            'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
            'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
            'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
            'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
            'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
            'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
            'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
            'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
            'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
            'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
            'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
            'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
            'TAC': 'Y', 'TAT': 'Y', 'TAA': '*', 'TAG': '*',
            'TGC': 'C', 'TGT': 'C', 'TGA': '*', 'TGG': 'W',
        }
        
        # Amino acid properties
        self.aa_properties = {
            'A': {'hydrophobicity': 1.8, 'charge': 0, 'polarity': 0, 'size': 'small'},
            'C': {'hydrophobicity': 2.5, 'charge': 0, 'polarity': 0, 'size': 'small'},
            'D': {'hydrophobicity': -3.5, 'charge': -1, 'polarity': 1, 'size': 'small'},
            'E': {'hydrophobicity': -3.5, 'charge': -1, 'polarity': 1, 'size': 'medium'},
            'F': {'hydrophobicity': 2.8, 'charge': 0, 'polarity': 0, 'size': 'large'},
            'G': {'hydrophobicity': -0.4, 'charge': 0, 'polarity': 0, 'size': 'small'},
            'H': {'hydrophobicity': -3.2, 'charge': 0.1, 'polarity': 1, 'size': 'medium'},
            'I': {'hydrophobicity': 4.5, 'charge': 0, 'polarity': 0, 'size': 'large'},
            'K': {'hydrophobicity': -3.9, 'charge': 1, 'polarity': 1, 'size': 'large'},
            'L': {'hydrophobicity': 3.8, 'charge': 0, 'polarity': 0, 'size': 'large'},
            'M': {'hydrophobicity': 1.9, 'charge': 0, 'polarity': 0, 'size': 'large'},
            'N': {'hydrophobicity': -3.5, 'charge': 0, 'polarity': 1, 'size': 'small'},
            'P': {'hydrophobicity': -1.6, 'charge': 0, 'polarity': 0, 'size': 'small'},
            'Q': {'hydrophobicity': -3.5, 'charge': 0, 'polarity': 1, 'size': 'medium'},
            'R': {'hydrophobicity': -4.5, 'charge': 1, 'polarity': 1, 'size': 'large'},
            'S': {'hydrophobicity': -0.8, 'charge': 0, 'polarity': 1, 'size': 'small'},
            'T': {'hydrophobicity': -0.7, 'charge': 0, 'polarity': 1, 'size': 'small'},
            'V': {'hydrophobicity': 4.2, 'charge': 0, 'polarity': 0, 'size': 'medium'},
            'W': {'hydrophobicity': -0.9, 'charge': 0, 'polarity': 0, 'size': 'large'},
            'Y': {'hydrophobicity': -1.3, 'charge': 0, 'polarity': 1, 'size': 'large'},
            '*': {'hydrophobicity': 0, 'charge': 0, 'polarity': 0, 'size': 'none'}
        }
        
        logger.info("ONI Bioinformatics initialized")
    
    def load_sequence(self, 
                     sequence: str,
                     name: str,
                     sequence_type: str = 'dna') -> Dict[str, Any]:
        """
        Load a biological sequence.
        
        Args:
            sequence: Sequence string
            name: Name for the sequence
            sequence_type: Type of sequence ('dna', 'rna', 'protein')
            
        Returns:
            Dict[str, Any]: Sequence information
        """
        try:
            # Clean sequence
            sequence = sequence.upper().strip()
            
            # Validate sequence
            if sequence_type == 'dna':
                valid_chars = set('ACGT')
                if not all(c in valid_chars for c in sequence):
                    invalid_chars = set(c for c in sequence if c not in valid_chars)
                    raise ValueError(f"Invalid characters in DNA sequence: {invalid_chars}")
            elif sequence_type == 'rna':
                valid_chars = set('ACGU')
                if not all(c in valid_chars for c in sequence):
                    invalid_chars = set(c for c in sequence if c not in valid_chars)
                    raise ValueError(f"Invalid characters in RNA sequence: {invalid_chars}")
            elif sequence_type == 'protein':
                valid_chars = set('ACDEFGHIKLMNPQRSTVWY*')
                if not all(c in valid_chars for c in sequence):
                    invalid_chars = set(c for c in sequence if c not in valid_chars)
                    raise ValueError(f"Invalid characters in protein sequence: {invalid_chars}")
            else:
                raise ValueError(f"Unsupported sequence type: {sequence_type}")
            
            # Create sequence dictionary
            seq_dict = {
                'name': name,
                'sequence': sequence,
                'type': sequence_type,
                'length': len(sequence)
            }
            
            # Store the sequence
            self.sequences[name] = seq_dict
            
            logger.info(f"Loaded {sequence_type} sequence '{name}' with length {len(sequence)}")
            return seq_dict
            
        except Exception as e:
            logger.error(f"Error loading sequence: {e}")
            raise
    
    def load_fasta(self, fasta_content: str) -> Dict[str, Dict[str, Any]]:
        """
        Load sequences from FASTA format.
        
        Args:
            fasta_content: FASTA format content
            
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of loaded sequences
        """
        try:
            # Split into entries
            entries = fasta_content.strip().split('>')
            
            # Remove empty entries
            entries = [entry for entry in entries if entry.strip()]
            
            loaded_sequences = {}
            
            for entry in entries:
                lines = entry.strip().split('\n')
                header = lines[0].strip()
                sequence = ''.join(lines[1:]).strip()
                
                # Extract name from header
                name = header.split()[0]
                
                # Determine sequence type
                if all(c in 'ACGT' for c in sequence.upper()):
                    seq_type = 'dna'
                elif all(c in 'ACGU' for c in sequence.upper()):
                    seq_type = 'rna'
                elif all(c in 'ACDEFGHIKLMNPQRSTVWY*' for c in sequence.upper()):
                    seq_type = 'protein'
                else:
                    seq_type = 'unknown'
                
                # Load sequence
                seq_dict = self.load_sequence(sequence, name, seq_type)
                
                # Add to result
                loaded_sequences[name] = seq_dict
            
            logger.info(f"Loaded {len(loaded_sequences)} sequences from FASTA content")
            return loaded_sequences
            
        except Exception as e:
            logger.error(f"Error loading FASTA content: {e}")
            raise
    
    def get_sequence(self, name: str) -> Dict[str, Any]:
        """
        Get a sequence by name.
        
        Args:
            name: Name of the sequence
            
        Returns:
            Dict[str, Any]: Sequence information
        """
        if name not in self.sequences:
            raise ValueError(f"Sequence '{name}' not found")
        
        return self.sequences[name]
    
    def list_sequences(self) -> Dict[str, Tuple[str, int]]:
        """
        List all available sequences.
        
        Returns:
            Dict[str, Tuple[str, int]]: Dictionary mapping sequence names to (type, length)
        """
        return {name: (seq['type'], seq['length']) for name, seq in self.sequences.items()}
    
    def transcribe(self, dna_name: str, rna_name: str = None) -> Dict[str, Any]:
        """
        Transcribe DNA to RNA.
        
        Args:
            dna_name: Name of the DNA sequence
            rna_name: Name for the RNA sequence (optional)
            
        Returns:
            Dict[str, Any]: RNA sequence information
        """
        try:
            if dna_name not in self.sequences:
                raise ValueError(f"Sequence '{dna_name}' not found")
            
            dna_seq = self.sequences[dna_name]
            
            if dna_seq['type'] != 'dna':
                raise ValueError(f"Sequence '{dna_name}' is not DNA")
            
            # Transcribe DNA to RNA
            rna_sequence = dna_seq['sequence'].replace('T', 'U')
            
            # Generate RNA name if not provided
            if rna_name is None:
                rna_name = f"{dna_name}_rna"
            
            # Create RNA sequence
            rna_seq = self.load_sequence(rna_sequence, rna_name, 'rna')
            
            logger.info(f"Transcribed DNA '{dna_name}' to RNA '{rna_name}'")
            return rna_seq
            
        except Exception as e:
            logger.error(f"Error transcribing DNA: {e}")
            raise
    
    def translate(self, 
                 nucleic_name: str,
                 protein_name: str = None,
                 start_codon: str = 'ATG',
                 stop_codons: List[str] = None) -> Dict[str, Any]:
        """
        Translate DNA/RNA to protein.
        
        Args:
            nucleic_name: Name of the DNA/RNA sequence
            protein_name: Name for the protein sequence (optional)
            start_codon: Start codon (default: ATG)
            stop_codons: List of stop codons (optional)
            
        Returns:
            Dict[str, Any]: Protein sequence information
        """
        try:
            if nucleic_name not in self.sequences:
                raise ValueError(f"Sequence '{nucleic_name}' not found")
            
            nucleic_seq = self.sequences[nucleic_name]
            
            if nucleic_seq['type'] not in ['dna', 'rna']:
                raise ValueError(f"Sequence '{nucleic_name}' is not DNA or RNA")
            
            # Set default stop codons
            if stop_codons is None:
                if nucleic_seq['type'] == 'dna':
                    stop_codons = ['TAA', 'TAG', 'TGA']
                else:  # RNA
                    stop_codons = ['UAA', 'UAG', 'UGA']
            
            # Convert RNA to DNA for consistent processing
            sequence = nucleic_seq['sequence']
            if nucleic_seq['type'] == 'rna':
                sequence = sequence.replace('U', 'T')
                start_codon = start_codon.replace('U', 'T')
                stop_codons = [codon.replace('U', 'T') for codon in stop_codons]
            
            # Find start codon
            start_pos = sequence.find(start_codon)
            if start_pos == -1:
                logger.warning(f"Start codon '{start_codon}' not found in sequence '{nucleic_name}'")
                start_pos = 0
            
            # Translate to protein
            protein_sequence = ""
            for i in range(start_pos, len(sequence) - 2, 3):
                codon = sequence[i:i+3]
                
                # Check for stop codon
                if codon in stop_codons:
                    protein_sequence += '*'
                    break
                
                # Translate codon
                if len(codon) == 3:
                    aa = self.genetic_code.get(codon, 'X')
                    protein_sequence += aa
            
            # Generate protein name if not provided
            if protein_name is None:
                protein_name = f"{nucleic_name}_protein"
            
            # Create protein sequence
            protein_seq = self.load_sequence(protein_sequence, protein_name, 'protein')
            
            logger.info(f"Translated {nucleic_seq['type']} '{nucleic_name}' to protein '{protein_name}'")
            return protein_seq
            
        except Exception as e:
            logger.error(f"Error translating sequence: {e}")
            raise
    
    def reverse_complement(self, dna_name: str, output_name: str = None) -> Dict[str, Any]:
        """
        Calculate the reverse complement of a DNA sequence.
        
        Args:
            dna_name: Name of the DNA sequence
            output_name: Name for the output sequence (optional)
            
        Returns:
            Dict[str, Any]: Reverse complement sequence information
        """
        try:
            if dna_name not in self.sequences:
                raise ValueError(f"Sequence '{dna_name}' not found")
            
            dna_seq = self.sequences[dna_name]
            
            if dna_seq['type'] != 'dna':
                raise ValueError(f"Sequence '{dna_name}' is not DNA")
            
            # Calculate reverse complement
            complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
            reverse_complement_sequence = ''.join(complement_map.get(base, base) for base in reversed(dna_seq['sequence']))
            
            # Generate output name if not provided
            if output_name is None:
                output_name = f"{dna_name}_revcomp"
            
            # Create reverse complement sequence
            revcomp_seq = self.load_sequence(reverse_complement_sequence, output_name, 'dna')
            
            logger.info(f"Calculated reverse complement of DNA '{dna_name}' as '{output_name}'")
            return revcomp_seq
            
        except Exception as e:
            logger.error(f"Error calculating reverse complement: {e}")
            raise
    
    def find_orfs(self, 
                 nucleic_name: str,
                 min_length: int = 100,
                 start_codon: str = 'ATG',
                 stop_codons: List[str] = None) -> Dict[str, Any]:
        """
        Find open reading frames (ORFs) in a DNA/RNA sequence.
        
        Args:
            nucleic_name: Name of the DNA/RNA sequence
            min_length: Minimum ORF length in nucleotides
            start_codon: Start codon (default: ATG)
            stop_codons: List of stop codons (optional)
            
        Returns:
            Dict[str, Any]: ORF analysis results
        """
        try:
            if nucleic_name not in self.sequences:
                raise ValueError(f"Sequence '{nucleic_name}' not found")
            
            nucleic_seq = self.sequences[nucleic_name]
            
            if nucleic_seq['type'] not in ['dna', 'rna']:
                raise ValueError(f"Sequence '{nucleic_name}' is not DNA or RNA")
            
            # Set default stop codons
            if stop_codons is None:
                if nucleic_seq['type'] == 'dna':
                    stop_codons = ['TAA', 'TAG', 'TGA']
                else:  # RNA
                    stop_codons = ['UAA', 'UAG', 'UGA']
            
            # Convert RNA to DNA for consistent processing
            sequence = nucleic_seq['sequence']
            if nucleic_seq['type'] == 'rna':
                sequence = sequence.replace('U', 'T')
                start_codon = start_codon.replace('U', 'T')
                stop_codons = [codon.replace('U', 'T') for codon in stop_codons]
            
            # Find ORFs in all reading frames
            orfs = []
            
            for frame in range(3):
                # Find all start codons
                start_positions = []
                for i in range(frame, len(sequence) - 2, 3):
                    if sequence[i:i+3] == start_codon:
                        start_positions.append(i)
                
                # Find ORFs
                for start_pos in start_positions:
                    # Find next stop codon
                    stop_pos = None
                    for i in range(start_pos, len(sequence) - 2, 3):
                        if sequence[i:i+3] in stop_codons:
                            stop_pos = i + 3
                            break
                    
                    # If no stop codon found, use end of sequence
                    if stop_pos is None:
                        stop_pos = len(sequence) - (len(sequence) - start_pos) % 3
                    
                    # Calculate ORF length
                    orf_length = stop_pos - start_pos
                    
                    # Check minimum length
                    if orf_length >= min_length:
                        # Extract ORF sequence
                        orf_sequence = sequence[start_pos:stop_pos]
                        
                        # Translate ORF
                        protein_sequence = ""
                        for i in range(0, len(orf_sequence) - 2, 3):
                            codon = orf_sequence[i:i+3]
                            if len(codon) == 3:
                                aa = self.genetic_code.get(codon, 'X')
                                protein_sequence += aa
                        
                        # Add ORF to list
                        orfs.append({
                            'start': start_pos,
                            'end': stop_pos,
                            'frame': frame + 1,
                            'length': orf_length,
                            'sequence': orf_sequence,
                            'protein': protein_sequence
                        })
            
            # Sort ORFs by length (descending)
            orfs.sort(key=lambda x: x['length'], reverse=True)
            
            # Create result
            result = {
                'sequence_name': nucleic_name,
                'sequence_type': nucleic_seq['type'],
                'sequence_length': nucleic_seq['length'],
                'min_length': min_length,
                'start_codon': start_codon,
                'stop_codons': stop_codons,
                'orfs': orfs,
                'orf_count': len(orfs)
            }
            
            # Store the results
            result_name = f"{nucleic_name}_orfs"
            self.analysis_results[result_name] = result
            
            logger.info(f"Found {len(orfs)} ORFs in sequence '{nucleic_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Error finding ORFs: {e}")
            raise
    
    def calculate_gc_content(self, 
                            nucleic_name: str,
                            window_size: int = None) -> Dict[str, Any]:
        """
        Calculate GC content of a DNA/RNA sequence.
        
        Args:
            nucleic_name: Name of the DNA/RNA sequence
            window_size: Window size for sliding window analysis (optional)
            
        Returns:
            Dict[str, Any]: GC content analysis results
        """
        try:
            if nucleic_name not in self.sequences:
                raise ValueError(f"Sequence '{nucleic_name}' not found")
            
            nucleic_seq = self.sequences[nucleic_name]
            
            if nucleic_seq['type'] not in ['dna', 'rna']:
                raise ValueError(f"Sequence '{nucleic_name}' is not DNA or RNA")
            
            sequence = nucleic_seq['sequence']
            
            # Calculate overall GC content
            gc_count = sequence.count('G') + sequence.count('C')
            if nucleic_seq['type'] == 'rna':
                gc_count = sequence.count('G') + sequence.count('C')
            
            gc_content = gc_count / len(sequence) if len(sequence) > 0 else 0
            
            # Calculate GC content in sliding windows if requested
            window_gc = None
            if window_size is not None:
                window_gc = []
                for i in range(0, len(sequence) - window_size + 1):
                    window = sequence[i:i+window_size]
                    window_gc_count = window.count('G') + window.count('C')
                    window_gc.append(window_gc_count / window_size)
            
            # Create result
            result = {
                'sequence_name': nucleic_name,
                'sequence_type': nucleic_seq['type'],
                'sequence_length': nucleic_seq['length'],
                'gc_count': gc_count,
                'gc_content': float(gc_content),
                'window_size': window_size,
                'window_gc': window_gc
            }
            
            # Store the results
            result_name = f"{nucleic_name}_gc_content"
            self.analysis_results[result_name] = result
            
            logger.info(f"Calculated GC content for sequence '{nucleic_name}': {gc_content:.4f}")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating GC content: {e}")
            raise
    
    def plot_gc_content(self, 
                       gc_result: Dict[str, Any],
                       figsize: Tuple[int, int] = (10, 6),
                       return_base64: bool = True) -> Union[str, plt.Figure]:
        """
        Plot GC content analysis results.
        
        Args:
            gc_result: GC content analysis results
            figsize: Figure size as (width, height)
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Union[str, plt.Figure]: Base64-encoded image or matplotlib Figure
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Check if window analysis was performed
            if gc_result['window_gc'] is not None:
                # Plot sliding window GC content
                window_size = gc_result['window_size']
                x = np.arange(len(gc_result['window_gc'])) + window_size // 2
                ax.plot(x, gc_result['window_gc'])
                
                # Add horizontal line for overall GC content
                ax.axhline(y=gc_result['gc_content'], color='r', linestyle='--', alpha=0.7, label='Overall GC content')
                
                # Set labels and title
                ax.set_xlabel('Position')
                ax.set_ylabel('GC content')
                ax.set_title(f"GC content of {gc_result['sequence_name']} (window size: {window_size})")
                
                # Add legend
                ax.legend()
                
            else:
                # Plot overall GC content as bar
                ax.bar(['GC content'], [gc_result['gc_content']])
                
                # Set labels and title
                ax.set_ylabel('GC content')
                ax.set_title(f"GC content of {gc_result['sequence_name']}")
                
                # Set y-axis limits
                ax.set_ylim(0, 1)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Tight layout
            fig.tight_layout()
            
            # Increment figure counter
            self.figure_counter += 1
            
            if return_base64:
                # Convert to base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return img_str
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting GC content: {e}")
            raise
    
    def calculate_nucleotide_frequency(self, nucleic_name: str) -> Dict[str, Any]:
        """
        Calculate nucleotide frequency in a DNA/RNA sequence.
        
        Args:
            nucleic_name: Name of the DNA/RNA sequence
            
        Returns:
            Dict[str, Any]: Nucleotide frequency analysis results
        """
        try:
            if nucleic_name not in self.sequences:
                raise ValueError(f"Sequence '{nucleic_name}' not found")
            
            nucleic_seq = self.sequences[nucleic_name]
            
            if nucleic_seq['type'] not in ['dna', 'rna']:
                raise ValueError(f"Sequence '{nucleic_name}' is not DNA or RNA")
            
            sequence = nucleic_seq['sequence']
            
            # Calculate nucleotide frequencies
            total_length = len(sequence)
            frequencies = {}
            
            if nucleic_seq['type'] == 'dna':
                bases = ['A', 'C', 'G', 'T']
            else:  # RNA
                bases = ['A', 'C', 'G', 'U']
            
            for base in bases:
                count = sequence.count(base)
                frequencies[base] = count / total_length if total_length > 0 else 0
            
            # Calculate dinucleotide frequencies
            dinucleotide_counts = {}
            for i in range(len(sequence) - 1):
                dinucleotide = sequence[i:i+2]
                dinucleotide_counts[dinucleotide] = dinucleotide_counts.get(dinucleotide, 0) + 1
            
            dinucleotide_frequencies = {}
            for dinucleotide, count in dinucleotide_counts.items():
                dinucleotide_frequencies[dinucleotide] = count / (total_length - 1) if total_length > 1 else 0
            
            # Create result
            result = {
                'sequence_name': nucleic_name,
                'sequence_type': nucleic_seq['type'],
                'sequence_length': nucleic_seq['length'],
                'nucleotide_frequencies': frequencies,
                'dinucleotide_frequencies': dinucleotide_frequencies
            }
            
            # Store the results
            result_name = f"{nucleic_name}_nucleotide_freq"
            self.analysis_results[result_name] = result
            
            logger.info(f"Calculated nucleotide frequencies for sequence '{nucleic_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating nucleotide frequencies: {e}")
            raise
    
    def plot_nucleotide_frequency(self, 
                                 freq_result: Dict[str, Any],
                                 plot_type: str = 'nucleotide',
                                 figsize: Tuple[int, int] = (10, 6),
                                 return_base64: bool = True) -> Union[str, plt.Figure]:
        """
        Plot nucleotide frequency analysis results.
        
        Args:
            freq_result: Nucleotide frequency analysis results
            plot_type: Type of plot ('nucleotide', 'dinucleotide')
            figsize: Figure size as (width, height)
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Union[str, plt.Figure]: Base64-encoded image or matplotlib Figure
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            if plot_type == 'nucleotide':
                # Plot nucleotide frequencies
                frequencies = freq_result['nucleotide_frequencies']
                
                # Sort by nucleotide
                sorted_bases = sorted(frequencies.keys())
                sorted_freqs = [frequencies[base] for base in sorted_bases]
                
                # Plot bar chart
                ax.bar(sorted_bases, sorted_freqs)
                
                # Set labels and title
                ax.set_xlabel('Nucleotide')
                ax.set_ylabel('Frequency')
                ax.set_title(f"Nucleotide frequencies of {freq_result['sequence_name']}")
                
                # Set y-axis limits
                ax.set_ylim(0, 1)
                
            elif plot_type == 'dinucleotide':
                # Plot dinucleotide frequencies
                frequencies = freq_result['dinucleotide_frequencies']
                
                # Sort by dinucleotide
                sorted_dinucleotides = sorted(frequencies.keys())
                sorted_freqs = [frequencies[dinucleotide] for dinucleotide in sorted_dinucleotides]
                
                # Plot bar chart
                ax.bar(sorted_dinucleotides, sorted_freqs)
                
                # Set labels and title
                ax.set_xlabel('Dinucleotide')
                ax.set_ylabel('Frequency')
                ax.set_title(f"Dinucleotide frequencies of {freq_result['sequence_name']}")
                
                # Set y-axis limits
                ax.set_ylim(0, max(sorted_freqs) * 1.1)
                
                # Rotate x-axis labels for readability
                plt.xticks(rotation=90)
                
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Tight layout
            fig.tight_layout()
            
            # Increment figure counter
            self.figure_counter += 1
            
            if return_base64:
                # Convert to base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return img_str
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting nucleotide frequencies: {e}")
            raise
    
    def calculate_protein_properties(self, protein_name: str) -> Dict[str, Any]:
        """
        Calculate properties of a protein sequence.
        
        Args:
            protein_name: Name of the protein sequence
            
        Returns:
            Dict[str, Any]: Protein property analysis results
        """
        try:
            if protein_name not in self.sequences:
                raise ValueError(f"Sequence '{protein_name}' not found")
            
            protein_seq = self.sequences[protein_name]
            
            if protein_seq['type'] != 'protein':
                raise ValueError(f"Sequence '{protein_name}' is not a protein")
            
            sequence = protein_seq['sequence']
            
            # Calculate amino acid frequencies
            aa_counts = {}
            for aa in sequence:
                aa_counts[aa] = aa_counts.get(aa, 0) + 1
            
            aa_frequencies = {}
            for aa, count in aa_counts.items():
                aa_frequencies[aa] = count / len(sequence) if len(sequence) > 0 else 0
            
            # Calculate average properties
            hydrophobicity = 0
            charge = 0
            polarity_count = 0
            
            size_counts = {'small': 0, 'medium': 0, 'large': 0, 'none': 0}
            
            for aa in sequence:
                if aa in self.aa_properties:
                    props = self.aa_properties[aa]
                    hydrophobicity += props['hydrophobicity']
                    charge += props['charge']
                    polarity_count += props['polarity']
                    size_counts[props['size']] += 1
            
            avg_hydrophobicity = hydrophobicity / len(sequence) if len(sequence) > 0 else 0
            avg_charge = charge / len(sequence) if len(sequence) > 0 else 0
            polarity_fraction = polarity_count / len(sequence) if len(sequence) > 0 else 0
            
            # Calculate size distribution
            size_distribution = {}
            for size, count in size_counts.items():
                size_distribution[size] = count / len(sequence) if len(sequence) > 0 else 0
            
            # Calculate molecular weight (approximate)
            aa_weights = {
                'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
                'G': 75.07, 'H': 155.16, 'I': 131.17, 'K': 146.19, 'L': 131.17,
                'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
                'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19,
                '*': 0
            }
            
            molecular_weight = sum(aa_weights.get(aa, 0) for aa in sequence)
            
            # Create result
            result = {
                'sequence_name': protein_name,
                'sequence_length': protein_seq['length'],
                'aa_frequencies': aa_frequencies,
                'avg_hydrophobicity': float(avg_hydrophobicity),
                'avg_charge': float(avg_charge),
                'polarity_fraction': float(polarity_fraction),
                'size_distribution': size_distribution,
                'molecular_weight': float(molecular_weight)
            }
            
            # Store the results
            result_name = f"{protein_name}_properties"
            self.analysis_results[result_name] = result
            
            logger.info(f"Calculated properties for protein '{protein_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating protein properties: {e}")
            raise
    
    def plot_protein_properties(self, 
                               prop_result: Dict[str, Any],
                               plot_type: str = 'hydrophobicity',
                               window_size: int = 9,
                               figsize: Tuple[int, int] = (10, 6),
                               return_base64: bool = True) -> Union[str, plt.Figure]:
        """
        Plot protein property analysis results.
        
        Args:
            prop_result: Protein property analysis results
            plot_type: Type of plot ('hydrophobicity', 'aa_composition', 'size_distribution')
            window_size: Window size for hydrophobicity plot
            figsize: Figure size as (width, height)
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Union[str, plt.Figure]: Base64-encoded image or matplotlib Figure
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            if plot_type == 'hydrophobicity':
                # Get protein sequence
                protein_name = prop_result['sequence_name']
                if protein_name not in self.sequences:
                    raise ValueError(f"Sequence '{protein_name}' not found")
                
                protein_seq = self.sequences[protein_name]
                sequence = protein_seq['sequence']
                
                # Calculate hydrophobicity profile
                hydrophobicity_values = []
                for aa in sequence:
                    if aa in self.aa_properties:
                        hydrophobicity_values.append(self.aa_properties[aa]['hydrophobicity'])
                    else:
                        hydrophobicity_values.append(0)
                
                # Calculate sliding window average
                window_hydrophobicity = []
                half_window = window_size // 2
                
                for i in range(len(sequence)):
                    start = max(0, i - half_window)
                    end = min(len(sequence), i + half_window + 1)
                    window_avg = sum(hydrophobicity_values[start:end]) / (end - start)
                    window_hydrophobicity.append(window_avg)
                
                # Plot hydrophobicity profile
                ax.plot(range(1, len(sequence) + 1), window_hydrophobicity)
                
                # Add horizontal line at y=0
                ax.axhline(y=0, color='r', linestyle='--', alpha=0.7)
                
                # Set labels and title
                ax.set_xlabel('Residue position')
                ax.set_ylabel('Hydrophobicity')
                ax.set_title(f"Hydrophobicity profile of {protein_name} (window size: {window_size})")
                
            elif plot_type == 'aa_composition':
                # Plot amino acid composition
                aa_frequencies = prop_result['aa_frequencies']
                
                # Sort by amino acid
                sorted_aa = sorted(aa_frequencies.keys())
                sorted_freqs = [aa_frequencies[aa] for aa in sorted_aa]
                
                # Plot bar chart
                ax.bar(sorted_aa, sorted_freqs)
                
                # Set labels and title
                ax.set_xlabel('Amino acid')
                ax.set_ylabel('Frequency')
                ax.set_title(f"Amino acid composition of {prop_result['sequence_name']}")
                
                # Set y-axis limits
                ax.set_ylim(0, max(sorted_freqs) * 1.1)
                
            elif plot_type == 'size_distribution':
                # Plot size distribution
                size_distribution = prop_result['size_distribution']
                
                # Sort by size category
                size_order = ['small', 'medium', 'large', 'none']
                sorted_sizes = [size for size in size_order if size in size_distribution]
                sorted_freqs = [size_distribution[size] for size in sorted_sizes]
                
                # Plot bar chart
                ax.bar(sorted_sizes, sorted_freqs)
                
                # Set labels and title
                ax.set_xlabel('Residue size')
                ax.set_ylabel('Fraction')
                ax.set_title(f"Residue size distribution of {prop_result['sequence_name']}")
                
                # Set y-axis limits
                ax.set_ylim(0, 1)
                
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Tight layout
            fig.tight_layout()
            
            # Increment figure counter
            self.figure_counter += 1
            
            if return_base64:
                # Convert to base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return img_str
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting protein properties: {e}")
            raise
    
    def align_sequences(self, 
                       seq_name1: str,
                       seq_name2: str,
                       alignment_name: str = None,
                       gap_penalty: float = -1,
                       match_score: float = 1,
                       mismatch_penalty: float = -1) -> Dict[str, Any]:
        """
        Align two sequences using the Needleman-Wunsch algorithm.
        
        Args:
            seq_name1: Name of the first sequence
            seq_name2: Name of the second sequence
            alignment_name: Name for the alignment (optional)
            gap_penalty: Gap penalty
            match_score: Match score
            mismatch_penalty: Mismatch penalty
            
        Returns:
            Dict[str, Any]: Alignment results
        """
        try:
            if seq_name1 not in self.sequences:
                raise ValueError(f"Sequence '{seq_name1}' not found")
            
            if seq_name2 not in self.sequences:
                raise ValueError(f"Sequence '{seq_name2}' not found")
            
            seq1 = self.sequences[seq_name1]
            seq2 = self.sequences[seq_name2]
            
            # Check if sequences are of the same type
            if seq1['type'] != seq2['type']:
                raise ValueError(f"Sequences '{seq_name1}' and '{seq_name2}' are of different types")
            
            # Get sequences
            sequence1 = seq1['sequence']
            sequence2 = seq2['sequence']
            
            # Perform global alignment
            aligned_seq1, aligned_seq2, score = self._needleman_wunsch(
                sequence1, sequence2, gap_penalty, match_score, mismatch_penalty
            )
            
            # Calculate alignment statistics
            matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b)
            mismatches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a != b and a != '-' and b != '-')
            gaps = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == '-' or b == '-')
            
            alignment_length = len(aligned_seq1)
            identity = matches / alignment_length if alignment_length > 0 else 0
            
            # Generate alignment name if not provided
            if alignment_name is None:
                alignment_name = f"{seq_name1}_{seq_name2}_alignment"
            
            # Create alignment result
            alignment = {
                'name': alignment_name,
                'seq_name1': seq_name1,
                'seq_name2': seq_name2,
                'seq_type': seq1['type'],
                'aligned_seq1': aligned_seq1,
                'aligned_seq2': aligned_seq2,
                'score': score,
                'length': alignment_length,
                'matches': matches,
                'mismatches': mismatches,
                'gaps': gaps,
                'identity': identity
            }
            
            # Store the alignment
            self.alignments[alignment_name] = alignment
            
            logger.info(f"Aligned sequences '{seq_name1}' and '{seq_name2}' with identity {identity:.4f}")
            return alignment
            
        except Exception as e:
            logger.error(f"Error aligning sequences: {e}")
            raise
    
    def _needleman_wunsch(self, 
                         seq1: str,
                         seq2: str,
                         gap_penalty: float = -1,
                         match_score: float = 1,
                         mismatch_penalty: float = -1) -> Tuple[str, str, float]:
        """
        Implement the Needleman-Wunsch algorithm for global sequence alignment.
        
        Args:
            seq1: First sequence
            seq2: Second sequence
            gap_penalty: Gap penalty
            match_score: Match score
            mismatch_penalty: Mismatch penalty
            
        Returns:
            Tuple[str, str, float]: Aligned sequences and alignment score
        """
        # Initialize score matrix
        n = len(seq1)
        m = len(seq2)
        
        score_matrix = np.zeros((n + 1, m + 1))
        
        # Initialize first row and column with gap penalties
        for i in range(n + 1):
            score_matrix[i, 0] = i * gap_penalty
        
        for j in range(m + 1):
            score_matrix[0, j] = j * gap_penalty
        
        # Fill the score matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                match = score_matrix[i-1, j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty)
                delete = score_matrix[i-1, j] + gap_penalty
                insert = score_matrix[i, j-1] + gap_penalty
                
                score_matrix[i, j] = max(match, delete, insert)
        
        # Traceback to find the alignment
        aligned_seq1 = ""
        aligned_seq2 = ""
        
        i, j = n, m
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and score_matrix[i, j] == score_matrix[i-1, j-1] + (match_score if seq1[i-1] == seq2[j-1] else mismatch_penalty):
                aligned_seq1 = seq1[i-1] + aligned_seq1
                aligned_seq2 = seq2[j-1] + aligned_seq2
                i -= 1
                j -= 1
            elif i > 0 and score_matrix[i, j] == score_matrix[i-1, j] + gap_penalty:
                aligned_seq1 = seq1[i-1] + aligned_seq1
                aligned_seq2 = '-' + aligned_seq2
                i -= 1
            else:
                aligned_seq1 = '-' + aligned_seq1
                aligned_seq2 = seq2[j-1] + aligned_seq2
                j -= 1
        
        return aligned_seq1, aligned_seq2, score_matrix[n, m]
    
    def plot_alignment(self, 
                      alignment_name: str,
                      figsize: Tuple[int, int] = (10, 6),
                      return_base64: bool = True) -> Union[str, plt.Figure]:
        """
        Plot a sequence alignment.
        
        Args:
            alignment_name: Name of the alignment
            figsize: Figure size as (width, height)
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Union[str, plt.Figure]: Base64-encoded image or matplotlib Figure
        """
        try:
            if alignment_name not in self.alignments:
                raise ValueError(f"Alignment '{alignment_name}' not found")
            
            alignment = self.alignments[alignment_name]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Hide axes
            ax.axis('off')
            
            # Get alignment data
            aligned_seq1 = alignment['aligned_seq1']
            aligned_seq2 = alignment['aligned_seq2']
            seq_name1 = alignment['seq_name1']
            seq_name2 = alignment['seq_name2']
            
            # Calculate how many characters to display per line
            chars_per_line = 80
            
            # Split alignment into lines
            for i in range(0, len(aligned_seq1), chars_per_line):
                # Get segment of alignment
                segment1 = aligned_seq1[i:i+chars_per_line]
                segment2 = aligned_seq2[i:i+chars_per_line]
                
                # Create match line
                match_line = ''.join('|' if a == b else ' ' for a, b in zip(segment1, segment2))
                
                # Calculate y-position
                y_pos = 1.0 - (i / len(aligned_seq1)) * 0.8
                
                # Add text
                ax.text(0.05, y_pos, f"{seq_name1}: {segment1}", fontfamily='monospace')
                ax.text(0.05, y_pos - 0.02, f"{' ' * (len(seq_name1) + 2)}{match_line}", fontfamily='monospace')
                ax.text(0.05, y_pos - 0.04, f"{seq_name2}: {segment2}", fontfamily='monospace')
            
            # Add alignment statistics
            stats_text = (
                f"Alignment length: {alignment['length']}\n"
                f"Matches: {alignment['matches']}\n"
                f"Mismatches: {alignment['mismatches']}\n"
                f"Gaps: {alignment['gaps']}\n"
                f"Identity: {alignment['identity']:.2%}"
            )
            
            ax.text(0.7, 0.9, stats_text, transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8))
            
            # Set title
            ax.set_title(f"Alignment of {seq_name1} and {seq_name2}")
            
            # Tight layout
            fig.tight_layout()
            
            # Increment figure counter
            self.figure_counter += 1
            
            if return_base64:
                # Convert to base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return img_str
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting alignment: {e}")
            raise
    
    def find_motifs(self, 
                   sequence_name: str,
                   motif: str,
                   allow_mismatches: int = 0) -> Dict[str, Any]:
        """
        Find sequence motifs.
        
        Args:
            sequence_name: Name of the sequence
            motif: Motif to search for
            allow_mismatches: Number of allowed mismatches
            
        Returns:
            Dict[str, Any]: Motif search results
        """
        try:
            if sequence_name not in self.sequences:
                raise ValueError(f"Sequence '{sequence_name}' not found")
            
            sequence_data = self.sequences[sequence_name]
            sequence = sequence_data['sequence']
            
            # Find motif occurrences
            motif_occurrences = []
            
            for i in range(len(sequence) - len(motif) + 1):
                # Extract subsequence
                subseq = sequence[i:i+len(motif)]
                
                # Count mismatches
                mismatches = sum(1 for a, b in zip(subseq, motif) if a != b)
                
                # Check if within allowed mismatches
                if mismatches <= allow_mismatches:
                    motif_occurrences.append({
                        'position': i,
                        'subsequence': subseq,
                        'mismatches': mismatches
                    })
            
            # Create result
            result = {
                'sequence_name': sequence_name,
                'sequence_type': sequence_data['type'],
                'motif': motif,
                'allow_mismatches': allow_mismatches,
                'occurrences': motif_occurrences,
                'count': len(motif_occurrences)
            }
            
            # Store the results
            result_name = f"{sequence_name}_motif_{motif}"
            self.analysis_results[result_name] = result
            
            logger.info(f"Found {len(motif_occurrences)} occurrences of motif '{motif}' in sequence '{sequence_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Error finding motifs: {e}")
            raise
    
    def plot_motif_locations(self, 
                            motif_result: Dict[str, Any],
                            figsize: Tuple[int, int] = (10, 6),
                            return_base64: bool = True) -> Union[str, plt.Figure]:
        """
        Plot motif locations in a sequence.
        
        Args:
            motif_result: Motif search results
            figsize: Figure size as (width, height)
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Union[str, plt.Figure]: Base64-encoded image or matplotlib Figure
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Get sequence information
            sequence_name = motif_result['sequence_name']
            if sequence_name not in self.sequences:
                raise ValueError(f"Sequence '{sequence_name}' not found")
            
            sequence_data = self.sequences[sequence_name]
            sequence_length = sequence_data['length']
            
            # Get motif information
            motif = motif_result['motif']
            occurrences = motif_result['occurrences']
            
            # Plot sequence as a line
            ax.plot([0, sequence_length], [0, 0], 'k-', linewidth=2)
            
            # Plot motif occurrences
            for i, occurrence in enumerate(occurrences):
                position = occurrence['position']
                mismatches = occurrence['mismatches']
                
                # Calculate color based on mismatches
                if mismatches == 0:
                    color = 'green'
                else:
                    # Gradient from green to red based on mismatches
                    color = (min(1, mismatches / 3), max(0, 1 - mismatches / 3), 0)
                
                # Plot motif as a rectangle
                rect = plt.Rectangle(
                    (position, -0.2),
                    len(motif),
                    0.4,
                    color=color,
                    alpha=0.7
                )
                ax.add_patch(rect)
            
            # Set labels and title
            ax.set_xlabel('Position')
            ax.set_title(f"Locations of motif '{motif}' in {sequence_name}")
            
            # Set y-axis limits and hide y-axis
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            
            # Set x-axis limits
            ax.set_xlim(-10, sequence_length + 10)
            
            # Add legend
            if motif_result['allow_mismatches'] > 0:
                # Create legend patches
                legend_patches = [
                    plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.7, label='Perfect match'),
                    plt.Rectangle((0, 0), 1, 1, color='orange', alpha=0.7, label='With mismatches')
                ]
                ax.legend(handles=legend_patches)
            
            # Tight layout
            fig.tight_layout()
            
            # Increment figure counter
            self.figure_counter += 1
            
            if return_base64:
                # Convert to base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return img_str
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting motif locations: {e}")
            raise
    
    def calculate_codon_usage(self, nucleic_name: str) -> Dict[str, Any]:
        """
        Calculate codon usage in a DNA/RNA sequence.
        
        Args:
            nucleic_name: Name of the DNA/RNA sequence
            
        Returns:
            Dict[str, Any]: Codon usage analysis results
        """
        try:
            if nucleic_name not in self.sequences:
                raise ValueError(f"Sequence '{nucleic_name}' not found")
            
            nucleic_seq = self.sequences[nucleic_name]
            
            if nucleic_seq['type'] not in ['dna', 'rna']:
                raise ValueError(f"Sequence '{nucleic_name}' is not DNA or RNA")
            
            sequence = nucleic_seq['sequence']
            
            # Convert RNA to DNA for consistent processing
            if nucleic_seq['type'] == 'rna':
                sequence = sequence.replace('U', 'T')
            
            # Count codons
            codon_counts = {}
            for i in range(0, len(sequence) - 2, 3):
                codon = sequence[i:i+3]
                if len(codon) == 3:
                    codon_counts[codon] = codon_counts.get(codon, 0) + 1
            
            # Calculate codon frequencies
            total_codons = sum(codon_counts.values())
            codon_frequencies = {}
            for codon, count in codon_counts.items():
                codon_frequencies[codon] = count / total_codons if total_codons > 0 else 0
            
            # Group codons by amino acid
            aa_codon_counts = {}
            for codon, count in codon_counts.items():
                aa = self.genetic_code.get(codon, 'X')
                if aa not in aa_codon_counts:
                    aa_codon_counts[aa] = {}
                aa_codon_counts[aa][codon] = count
            
            # Calculate relative synonymous codon usage (RSCU)
            rscu = {}
            for aa, codons in aa_codon_counts.items():
                total_aa_count = sum(codons.values())
                num_codons = len(codons)
                
                if total_aa_count > 0 and num_codons > 0:
                    expected_count = total_aa_count / num_codons
                    
                    for codon, count in codons.items():
                        rscu[codon] = count / expected_count
            
            # Create result
            result = {
                'sequence_name': nucleic_name,
                'sequence_type': nucleic_seq['type'],
                'sequence_length': nucleic_seq['length'],
                'codon_counts': codon_counts,
                'codon_frequencies': codon_frequencies,
                'aa_codon_counts': aa_codon_counts,
                'rscu': rscu
            }
            
            # Store the results
            result_name = f"{nucleic_name}_codon_usage"
            self.analysis_results[result_name] = result
            
            logger.info(f"Calculated codon usage for sequence '{nucleic_name}'")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating codon usage: {e}")
            raise
    
    def plot_codon_usage(self, 
                        codon_result: Dict[str, Any],
                        plot_type: str = 'frequency',
                        figsize: Tuple[int, int] = (12, 8),
                        return_base64: bool = True) -> Union[str, plt.Figure]:
        """
        Plot codon usage analysis results.
        
        Args:
            codon_result: Codon usage analysis results
            plot_type: Type of plot ('frequency', 'rscu', 'aa_distribution')
            figsize: Figure size as (width, height)
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Union[str, plt.Figure]: Base64-encoded image or matplotlib Figure
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            if plot_type == 'frequency':
                # Plot codon frequencies
                frequencies = codon_result['codon_frequencies']
                
                # Sort by codon
                sorted_codons = sorted(frequencies.keys())
                sorted_freqs = [frequencies[codon] for codon in sorted_codons]
                
                # Plot bar chart
                ax.bar(sorted_codons, sorted_freqs)
                
                # Set labels and title
                ax.set_xlabel('Codon')
                ax.set_ylabel('Frequency')
                ax.set_title(f"Codon usage frequencies of {codon_result['sequence_name']}")
                
                # Rotate x-axis labels for readability
                plt.xticks(rotation=90)
                
            elif plot_type == 'rscu':
                # Plot relative synonymous codon usage
                rscu = codon_result['rscu']
                
                # Sort by codon
                sorted_codons = sorted(rscu.keys())
                sorted_rscu = [rscu[codon] for codon in sorted_codons]
                
                # Plot bar chart
                ax.bar(sorted_codons, sorted_rscu)
                
                # Add horizontal line at RSCU = 1
                ax.axhline(y=1, color='r', linestyle='--', alpha=0.7)
                
                # Set labels and title
                ax.set_xlabel('Codon')
                ax.set_ylabel('RSCU')
                ax.set_title(f"Relative Synonymous Codon Usage of {codon_result['sequence_name']}")
                
                # Rotate x-axis labels for readability
                plt.xticks(rotation=90)
                
            elif plot_type == 'aa_distribution':
                # Plot amino acid distribution
                aa_codon_counts = codon_result['aa_codon_counts']
                
                # Calculate total count for each amino acid
                aa_counts = {aa: sum(codons.values()) for aa, codons in aa_codon_counts.items()}
                
                # Sort by amino acid
                sorted_aa = sorted(aa_counts.keys())
                sorted_counts = [aa_counts[aa] for aa in sorted_aa]
                
                # Plot bar chart
                ax.bar(sorted_aa, sorted_counts)
                
                # Set labels and title
                ax.set_xlabel('Amino acid')
                ax.set_ylabel('Count')
                ax.set_title(f"Amino acid distribution of {codon_result['sequence_name']}")
                
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Tight layout
            fig.tight_layout()
            
            # Increment figure counter
            self.figure_counter += 1
            
            if return_base64:
                # Convert to base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return img_str
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting codon usage: {e}")
            raise
    
    def run(self, command: str = None, **kwargs) -> Any:
        """
        Run a command or return help information.
        
        Args:
            command: Command to run (optional)
            **kwargs: Additional arguments
            
        Returns:
            Any: Command result or help information
        """
        if command is None:
            # Return help information
            return self.help()
        
        # Parse and execute command
        try:
            # Check if command is a method name
            if hasattr(self, command) and callable(getattr(self, command)):
                method = getattr(self, command)
                return method(**kwargs)
            
            # Otherwise, try to evaluate as a Python expression
            # This is potentially dangerous and should be used with caution
            return eval(command)
            
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            return f"Error: {str(e)}"
    
    def help(self) -> str:
        """Return help information about the ONI Bioinformatics module."""
        help_text = """
        ONI Bioinformatics - Analysis tools for biological data
        
        Available methods:
        - load_sequence(sequence, name, sequence_type): Load a biological sequence
        - load_fasta(fasta_content): Load sequences from FASTA format
        - get_sequence(name): Get a sequence by name
        - list_sequences(): List all available sequences
        - transcribe(dna_name, rna_name): Transcribe DNA to RNA
        - translate(nucleic_name, protein_name, ...): Translate DNA/RNA to protein
        - reverse_complement(dna_name, output_name): Calculate reverse complement
        - find_orfs(nucleic_name, ...): Find open reading frames
        - calculate_gc_content(nucleic_name, window_size): Calculate GC content
        - plot_gc_content(gc_result, ...): Plot GC content analysis results
        - calculate_nucleotide_frequency(nucleic_name): Calculate nucleotide frequency
        - plot_nucleotide_frequency(freq_result, ...): Plot nucleotide frequency results
        - calculate_protein_properties(protein_name): Calculate protein properties
        - plot_protein_properties(prop_result, ...): Plot protein property results
        - align_sequences(seq_name1, seq_name2, ...): Align two sequences
        - plot_alignment(alignment_name, ...): Plot a sequence alignment
        - find_motifs(sequence_name, motif, ...): Find sequence motifs
        - plot_motif_locations(motif_result, ...): Plot motif locations
        - calculate_codon_usage(nucleic_name): Calculate codon usage
        - plot_codon_usage(codon_result, ...): Plot codon usage results
        
        For more information on a specific method, use help(ONIBioinformatics.method_name)
        """
        return help_text
    
    def cleanup(self):
        """Clean up resources."""
        # Close all matplotlib figures
        plt.close('all')
        
        # Clear sequences and results
        self.sequences.clear()
        self.alignments.clear()
        self.analysis_results.clear()
        
        logger.info("ONI Bioinformatics cleaned up")

# Example usage
if __name__ == "__main__":
    bio = ONIBioinformatics()
    
    # Load a DNA sequence
    dna = bio.load_sequence("ATGGCTAGCAACGTGATCGTACGTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC", "example_dna", "dna")
    
    # Transcribe to RNA
    rna = bio.transcribe("example_dna", "example_rna")
    
    # Translate to protein
    protein = bio.translate("example_rna", "example_protein")
    
    # Calculate GC content
    gc_content = bio.calculate_gc_content("example_dna", window_size=10)
    
    # Plot GC content
    gc_plot = bio.plot_gc_content(gc_content)
    
    print(f"DNA sequence: {dna['sequence']}")
    print(f"RNA sequence: {rna['sequence']}")
    print(f"Protein sequence: {protein['sequence']}")
    print(f"GC content: {gc_content['gc_content']:.2f}")