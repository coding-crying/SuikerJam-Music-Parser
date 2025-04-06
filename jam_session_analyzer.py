#!/usr/bin/env python3
"""
Jam Session Analyzer

This script analyzes a long audio recording (like a jam session) to detect:
- Tempo changes
- Song boundaries
- Silence sections
- Clapping sections

It processes large files efficiently by streaming in chunks and outputs timestamps
of detected events.
"""

import os
import argparse
import json
import time
import warnings
import numpy as np
import librosa
import soundfile as sf
from datetime import timedelta

def format_timestamp(seconds):
    """Convert seconds to a human-readable timestamp format (HH:MM:SS)"""
    return str(timedelta(seconds=seconds)).split('.')[0]

def detect_events(audio_file, 
                 chunk_duration_sec=300,  # 5 minutes
                 overlap_sec=5,
                 sr=22050,
                 tempo_change_threshold_bpm=5.0,
                 boundary_peak_threshold=0.15,
                 silence_threshold_db=-50,
                 silence_min_duration_sec=2.0,
                 min_segment_length_sec=30,
                 clapping_min_duration_sec=1.0,
                 clapping_energy_percentile=20,
                 clapping_irregularity_threshold=0.6,
                 speech_min_duration_sec=1.0,
                 identify_song_boundaries_flag=True,
                 proximity_threshold_sec=30,
                 verbose=True):
    """
    Analyze an audio file to detect tempo changes and song boundaries.
    
    Parameters:
    -----------
    audio_file : str
        Path to the audio file to analyze
    chunk_duration_sec : float
        Duration of each processing chunk in seconds
    overlap_sec : float
        Overlap between consecutive chunks in seconds
    sr : int
        Target sample rate for analysis
    tempo_change_threshold_bpm : float
        Minimum difference in BPM to consider a tempo change
    boundary_peak_threshold : float
        Threshold for detecting segment boundaries from novelty curve
    silence_threshold_db : float
        Threshold in dB below which audio is considered silence
    silence_min_duration_sec : float
        Minimum duration of silence to be considered as a potential boundary
    min_segment_length_sec : float
        Minimum duration between consecutive song boundaries
    clapping_min_duration_sec : float
        Minimum duration for a clapping section to be detected
    clapping_energy_percentile : int
        Percentile of energy to consider for clapping detection (lower = more sensitive)
    clapping_irregularity_threshold : float
        Threshold for rhythmic irregularity to detect clapping (0-1, higher = more sensitive)
    speech_min_duration_sec : float
        Minimum duration for a speech section to be detected
    identify_song_boundaries_flag : bool
        Whether to perform high-confidence song boundary identification
    proximity_threshold_sec : float
        Maximum time difference for events to be considered related for song boundary detection
    verbose : bool
        Whether to print progress information
        
    Returns:
    --------
    list
        List of detected events with timestamps and types
    """
    
    # Validate input file
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    
    # Get file duration without loading the entire file
    with sf.SoundFile(audio_file) as f:
        file_duration_sec = len(f) / f.samplerate
        total_chunks = int(np.ceil(file_duration_sec / (chunk_duration_sec - overlap_sec)))
    
    if verbose:
        print(f"Analyzing: {audio_file}")
        print(f"Duration: {format_timestamp(file_duration_sec)}")
        print(f"Processing in {total_chunks} chunks of {chunk_duration_sec} seconds with {overlap_sec} seconds overlap")
        print("=" * 60)
    
    # Storage for results
    results = []
    last_tempo = None
    last_boundary_time = -min_segment_length_sec  # To allow a boundary at the very beginning
    
    # Process file in chunks
    chunk_start_time = 0.0
    chunk_idx = 0
    
    while chunk_start_time < file_duration_sec:
        chunk_duration = min(chunk_duration_sec, file_duration_sec - chunk_start_time)
        if chunk_duration < 10:  # Skip very short chunks (less than 10 seconds)
            break
            
        if verbose:
            print(f"Processing chunk {chunk_idx+1}/{total_chunks} - "
                  f"Time: {format_timestamp(chunk_start_time)} to {format_timestamp(chunk_start_time + chunk_duration)}")
        
        # Load chunk
        try:
            y, loaded_sr = librosa.load(audio_file, sr=sr, offset=chunk_start_time, duration=chunk_duration)
        except Exception as e:
            print(f"Error loading chunk at {format_timestamp(chunk_start_time)}: {e}")
            chunk_start_time += (chunk_duration_sec - overlap_sec)
            chunk_idx += 1
            continue
            
        # === TEMPO ANALYSIS ===
        tempo_results = detect_tempo_changes(y, sr, chunk_start_time, last_tempo, tempo_change_threshold_bpm)
        if tempo_results:
            last_tempo = tempo_results["tempo"]
            # Only add to results if there's an actual change
            if "change" in tempo_results:
                results.append({
                    "time": tempo_results["time"],
                    "type": "tempo_change",
                    "bpm": tempo_results["tempo"],
                    "prev_bpm": tempo_results["prev_tempo"]
                })
                if verbose:
                    print(f"Tempo change detected at {format_timestamp(tempo_results['time'])}: "
                          f"{tempo_results['prev_tempo']:.1f} → {tempo_results['tempo']:.1f} BPM")
        
        # === STRUCTURAL BOUNDARIES ===
        boundary_times = detect_boundaries(y, sr, chunk_start_time, boundary_peak_threshold)
        
        for boundary_time in boundary_times:
            # Avoid boundaries too close to each other
            if boundary_time - last_boundary_time >= min_segment_length_sec:
                results.append({
                    "time": boundary_time,
                    "type": "structural_boundary"
                })
                last_boundary_time = boundary_time
                if verbose:
                    print(f"Structural boundary detected at {format_timestamp(boundary_time)}")
        
        # === SILENCE DETECTION ===
        silence_times = detect_silences(y, sr, chunk_start_time, silence_threshold_db, silence_min_duration_sec)
        
        for silence_time, silence_end in silence_times:
            silence_duration = silence_end - silence_time
            # Avoid boundaries too close to each other
            if silence_time - last_boundary_time >= min_segment_length_sec:
                results.append({
                    "time": silence_time,
                    "type": "silence_boundary",
                    "duration": silence_duration
                })
                last_boundary_time = silence_time
                if verbose:
                    print(f"Silence boundary detected at {format_timestamp(silence_time)} "
                          f"(duration: {silence_duration:.1f}s)")
        
        # === CLAPPING DETECTION ===
        clapping_sections = detect_clapping(y, sr, chunk_start_time, 
                                           silence_threshold_db,
                                           clapping_min_duration_sec,
                                           clapping_energy_percentile,
                                           clapping_irregularity_threshold)
        
        for start_time, end_time, confidence in clapping_sections:
            clapping_duration = end_time - start_time
            results.append({
                "time": start_time,
                "end_time": end_time,
                "type": "clapping",
                "duration": clapping_duration,
                "confidence": confidence
            })
            if verbose:
                print(f"Clapping detected at {format_timestamp(start_time)} - "
                      f"{format_timestamp(end_time)} "
                      f"(duration: {clapping_duration:.1f}s, confidence: {confidence:.2f})")
        
        # === SPEECH DETECTION ===
        speech_sections = detect_speech(y, sr, chunk_start_time, 
                                       silence_threshold_db,
                                       speech_min_duration_sec)
        
        for start_time, end_time, confidence in speech_sections:
            speech_duration = end_time - start_time
            results.append({
                "time": start_time,
                "end_time": end_time,
                "type": "speech",
                "duration": speech_duration,
                "confidence": confidence
            })
        
        # Update for next chunk
        chunk_start_time += (chunk_duration_sec - overlap_sec)
        chunk_idx += 1
    
    # Sort results by time
    results.sort(key=lambda x: x["time"])
    
    # Identify high-confidence song boundaries if requested
    if identify_song_boundaries_flag and results:
        if verbose:
            print("\nIdentifying high-confidence song boundaries...")
            
        song_boundaries = identify_song_boundaries(results, proximity_threshold_sec)
        
        # Add song boundaries to results
        results.extend(song_boundaries)
        
        # Re-sort with song boundaries included
        results.sort(key=lambda x: x["time"])
        
        if verbose and song_boundaries:
            print(f"Identified {len(song_boundaries)} high-confidence song boundaries")
    
    return results

def detect_tempo_changes(y, sr, chunk_start_time, last_tempo, tempo_change_threshold):
    """
    Detect tempo and identify if there's a significant change from previous tempo.
    
    Returns a dict with tempo info, including change details if a change is detected.
    """
    if len(y) < sr * 3:  # Need at least 3 seconds for reliable tempo estimation
        return None
        
    # Suppress common warnings from librosa's beat tracker
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=512)
            
            # Check if tempo estimation worked
            if tempo is not None and len(beats) > 3:  # Need at least a few beats for confidence
                tempo = float(tempo)  # Ensure it's a float
                
                # If we have a previous tempo, check for a change
                if last_tempo is not None and abs(tempo - last_tempo) > tempo_change_threshold:
                    return {
                        "time": chunk_start_time,
                        "tempo": tempo,
                        "prev_tempo": last_tempo,
                        "change": abs(tempo - last_tempo)
                    }
                # No change or no previous tempo
                return {"time": chunk_start_time, "tempo": tempo}
        except Exception as e:
            # print(f"Warning: Beat tracking failed: {e}")
            pass
            
    return None

def detect_boundaries(y, sr, chunk_start_time, threshold):
    """
    Detect structural boundaries based on novelty in audio features.
    
    Uses a combination of MFCCs (timbre) and chroma (harmony) features.
    Returns a list of boundary timestamps.
    """
    if len(y) < sr * 3:  # Need at least 3 seconds
        return []
        
    hop_length = 512
    
    # Extract features - use both MFCCs for timbre and chroma for harmony
    try:
        # MFCCs for timbre
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
        
        # Chroma for harmonic content
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        
        # Combine features with weighting
        # Give more weight to MFCCs as they're often better for structure detection
        mfcc_weight, chroma_weight = 0.7, 0.3
        
        # Normalize each feature set before combining
        mfccs_normalized = librosa.util.normalize(mfccs, axis=1)
        chroma_normalized = librosa.util.normalize(chroma, axis=1)
        
        # Stack and weight
        combined_features = np.vstack([
            mfccs_normalized * mfcc_weight,
            chroma_normalized * chroma_weight
        ])
        
        # Calculate self-similarity matrix
        S = librosa.segment.recurrence_matrix(combined_features, 
                                             mode='affinity',
                                             width=5,  # Use several frames for smoothing
                                             sym=True)  # Force symmetry
        
        # Compute novelty curve using the diagonal of the recurrence matrix
        novelty = 1.0 - np.sum(np.diag(S, k=1)) / S.shape[0]
        kernel_size = 11
        novelty_smooth = np.convolve(novelty, 
                                     np.ones(kernel_size)/kernel_size, 
                                     mode='same')
        
        # Normalize novelty
        novelty_normalized = librosa.util.normalize(novelty_smooth)
        
        # Find peaks in novelty
        peaks = librosa.util.peak_pick(novelty_normalized,
                                      pre_max=20, post_max=20,  # Window size for local max
                                      pre_avg=20, post_avg=100,  # Window size for local avg
                                      delta=threshold,  # Minimum prominence
                                      wait=int(3 * sr / hop_length))  # Min distance (3 sec)
        
        # Convert frames to times and offset by chunk start
        peak_times = librosa.frames_to_time(peaks, sr=sr, hop_length=hop_length)
        return [chunk_start_time + t for t in peak_times]
        
    except Exception as e:
        # print(f"Warning: Structure detection failed: {e}")
        return []

def detect_silences(y, sr, chunk_start_time, threshold_db, min_duration_sec):
    """
    Detect periods of silence that could indicate song boundaries.
    
    Returns a list of (silence_start, silence_end) tuples in seconds.
    """
    if len(y) < sr * 2:  # Need at least 2 seconds
        return []
    
    # Compute RMS energy
    hop_length = 512
    frame_length = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Convert to dB scale
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Find frames below threshold
    silent_frames = np.where(rms_db < threshold_db)[0]
    
    # Group consecutive silent frames
    silence_boundaries = []
    min_frames = int(min_duration_sec * sr / hop_length)
    
    if len(silent_frames) > 0:
        # Group consecutive frame indices
        silent_regions = []
        region_start = silent_frames[0]
        current_region = [region_start]
        
        for i in range(1, len(silent_frames)):
            if silent_frames[i] == silent_frames[i-1] + 1:
                current_region.append(silent_frames[i])
            else:
                if len(current_region) >= min_frames:
                    silent_regions.append((current_region[0], current_region[-1]))
                region_start = silent_frames[i]
                current_region = [region_start]
                
        # Don't forget the last region
        if len(current_region) >= min_frames:
            silent_regions.append((current_region[0], current_region[-1]))
            
        # Convert frames to time
        for start_frame, end_frame in silent_regions:
            # We're interested in the start of silence as the boundary
            start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
            end_time = librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length)
            silence_boundaries.append((
                chunk_start_time + start_time,
                chunk_start_time + end_time
            ))
    
    return silence_boundaries

def detect_clapping(y, sr, chunk_start_time, silence_threshold_db, min_duration_sec, 
                   energy_percentile=20, irregularity_threshold=0.6):
    """
    Detect sections with applause/clapping based on audio characteristics.
    
    Clapping typically has:
    1. More high-frequency content than regular music
    2. More irregular transients (vs. rhythmic beats)
    3. Moderate energy levels (quieter than drumming, louder than silence)
    4. Less regular periodicity than music
    5. Less low-frequency energy compared to drums
    
    Parameters:
    -----------
    y : np.ndarray
        Audio signal
    sr : int
        Sample rate
    chunk_start_time : float
        Start time of this chunk in the overall recording
    silence_threshold_db : float
        Threshold below which audio is considered silence
    min_duration_sec : float
        Minimum duration for a clapping section
    energy_percentile : int
        Percentile of energy to consider for clapping detection
    irregularity_threshold : float
        Threshold for rhythmic irregularity (0-1, higher = more sensitive)
        
    Returns:
    --------
    list
        List of (start_time, end_time, confidence) tuples for detected clapping sections
    """
    if len(y) < sr * 3:  # Need at least 3 seconds
        return []
        
    hop_length = 512
    frame_length = 2048
    
    # === 1. Compute basic features ===
    
    # RMS energy for overall loudness
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Spectral centroid to look for high-frequency content (clapping has higher centroid)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    centroid_normalized = (centroid - np.min(centroid)) / (np.max(centroid) - np.min(centroid) + 1e-8)
    
    # Spectral flatness to detect noise-like quality (clapping is noisier than tonal sounds)
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    
    # Onset strength to detect transients
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    # Spectral contrast to pick up the difference between peaks and valleys
    # Higher values in mid-high bands are common for clapping
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_length)
    # Focus on mid-high frequency bands (bands 2-4 in a 6-band contrast)
    mid_high_contrast = np.mean(contrast[2:5, :], axis=0)
    
    # === 2. Compute ratio of high to low frequency content ===
    # Clapping has more high-frequency content than drums
    
    # Compute frequency band energy ratios
    # Get the STFT
    D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    
    # Compute power spectrogram
    S = np.abs(D)**2
    
    # Define frequency band ranges (in Hz)
    # Low (drums, bass): < 500 Hz
    # Mid (vocals, many instruments): 500-2000 Hz
    # High (cymbals, clapping): > 2000 Hz
    low_freq_idx = librosa.core.fft_frequencies(sr=sr, n_fft=frame_length) < 500
    high_freq_idx = librosa.core.fft_frequencies(sr=sr, n_fft=frame_length) > 2000
    
    # Sum energy in each band for each frame
    low_energy = np.sum(S[low_freq_idx, :], axis=0)
    high_energy = np.sum(S[high_freq_idx, :], axis=0)
    
    # Compute high to low ratio (larger value means more high freq content)
    # Add small constant to avoid division by zero
    high_low_ratio = high_energy / (low_energy + 1e-8)
    
    # Normalize to 0-1
    high_low_ratio_normalized = (high_low_ratio - np.min(high_low_ratio)) / (np.max(high_low_ratio) - np.min(high_low_ratio) + 1e-8)
    
    # === 3. Compute periodicity/regularity metrics with improved sensitivity ===
    
    # Use autocorrelation of onset envelope to measure periodicity
    onset_ac = librosa.autocorrelate(onset_env, max_size=sr//hop_length)
    onset_ac = onset_ac[:sr//(2*hop_length)]  # Look at reasonable tempo range
    
    # Normalize
    if np.max(onset_ac) > 0:
        onset_ac = onset_ac / np.max(onset_ac)
    
    # Calculate stronger rhythmic regularity metrics to better distinguish from drums
    
    # 1. Peak height relative to average (as before)
    if len(onset_ac) > 1:
        periodicity1 = np.max(onset_ac[1:]) / (np.mean(onset_ac[1:]) + 1e-8)
    else:
        periodicity1 = 1.0
        
    # 2. Count significant peaks in autocorrelation (many peaks = more regular rhythm like drums)
    if len(onset_ac) > 1:
        # Find peaks in autocorrelation
        ac_peaks = librosa.util.peak_pick(onset_ac, pre_max=1, post_max=1, pre_avg=1, post_avg=1, delta=0.1, wait=1)
        # Normalize by max possible peaks
        peak_count = len(ac_peaks) / (len(onset_ac) * 0.5)  # Scaling factor to get 0-1 range
        periodicity2 = min(peak_count, 1.0)  # Clip to 0-1 range
    else:
        periodicity2 = 1.0
    
    # 3. Compute variance of inter-onset intervals (IOIs)
    # Get onset frames
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, 
                                       backtrack=False,
                                       units='frames')
    
    # Calculate IOI variability if we have enough onsets
    if len(onsets) > 4:
        # Get inter-onset intervals
        iois = np.diff(onsets)
        # Compute coefficient of variation (normalized measure of dispersion)
        ioi_cv = np.std(iois) / (np.mean(iois) + 1e-8)
        # Higher values indicate more irregular timing (clapping)
        # Normalize to 0-1 with reasonable ceiling
        ioi_irregularity = min(ioi_cv / 2.0, 1.0)  # Cap at 1.0
    else:
        ioi_irregularity = 0.5  # Neutral value for insufficient data
    
    # Combine multiple rhythm irregularity metrics with weights
    rhythm_regularities = [periodicity1, periodicity2]
    rhythm_regularity = 0.5 * periodicity1 + 0.5 * periodicity2
    
    # Invert to get irregularity (higher = more irregular = more like clapping)
    irregularity = 1.0 - np.clip(rhythm_regularity, 0, 1)
    
    # Boost irregularity by IOI irregularity
    irregularity = 0.7 * irregularity + 0.3 * ioi_irregularity
    
    # === 4. Combine features to identify clapping sections ===
    
    # Energy range typical for clapping: louder than silence but typically quieter than music
    # We use percentiles to adapt to the overall energy of the recording
    min_energy_db = silence_threshold_db
    energy_threshold = np.percentile(rms_db, energy_percentile)
    
    # A frame is likely clapping if:
    # 1. Energy is in the right range (not too loud, not too quiet)
    # 2. Has high spectral centroid (clapping has strong high frequency content)
    # 3. Has high spectral flatness (noise-like quality)
    # 4. Has high high-to-low frequency ratio
    # 5. And the overall irregularity measure is high
    
    # For each frame, compute a "clapping likelihood" score (0-1)
    clapping_likelihood = np.zeros_like(rms_db)
    
    # Only consider frames with energy above silence but not too loud
    valid_energy_frames = (rms_db > min_energy_db) & (rms_db < energy_threshold)
    
    # Compute clapping score for each frame
    if np.any(valid_energy_frames):
        # Normalize each feature to 0-1 range for the valid frames
        centroid_score = centroid_normalized
        flatness_score = (flatness - np.min(flatness)) / (np.max(flatness) - np.min(flatness) + 1e-8)
        contrast_score = (mid_high_contrast - np.min(mid_high_contrast)) / (np.max(mid_high_contrast) - np.min(mid_high_contrast) + 1e-8)
        
        # Weighted combination of features
        clapping_likelihood[valid_energy_frames] = (
            0.25 * centroid_score[valid_energy_frames] +
            0.25 * flatness_score[valid_energy_frames] +
            0.15 * contrast_score[valid_energy_frames] +
            0.35 * high_low_ratio_normalized[valid_energy_frames]  # Added high-to-low freq ratio with higher weight
        )
        
        # Apply the global irregularity measure
        if irregularity > irregularity_threshold:
            # Boost the scores if we have high irregularity
            clapping_likelihood[valid_energy_frames] *= (1.0 + irregularity)
        else:
            # Strongly penalize regular patterns (like drums)
            clapping_likelihood[valid_energy_frames] *= (irregularity * 0.5)
        
        # Normalize again to 0-1
        if np.max(clapping_likelihood) > 0:
            clapping_likelihood = np.clip(clapping_likelihood / np.max(clapping_likelihood), 0, 1)
    
    # === 5. Identify continuous sections of clapping ===
    
    # Apply a threshold to the likelihood score
    clapping_threshold = 0.6  # Increased from 0.5 to be more strict
    potential_clapping_frames = np.where(clapping_likelihood > clapping_threshold)[0]
    
    # Group consecutive frames
    clapping_sections = []
    min_frames = int(min_duration_sec * sr / hop_length)
    
    if len(potential_clapping_frames) > 0:
        # Group consecutive indices
        region_start = potential_clapping_frames[0]
        current_region = [region_start]
        
        for i in range(1, len(potential_clapping_frames)):
            # Allow small gaps (up to 1 second) to account for irregularity in clapping
            max_gap = int(1.0 * sr / hop_length)
            if potential_clapping_frames[i] <= potential_clapping_frames[i-1] + max_gap:
                current_region.append(potential_clapping_frames[i])
            else:
                if len(current_region) >= min_frames:
                    avg_likelihood = np.mean(clapping_likelihood[current_region])
                    clapping_sections.append((
                        current_region[0], 
                        current_region[-1],
                        avg_likelihood
                    ))
                region_start = potential_clapping_frames[i]
                current_region = [region_start]
        
        # Don't forget the last region
        if len(current_region) >= min_frames:
            avg_likelihood = np.mean(clapping_likelihood[current_region])
            clapping_sections.append((
                current_region[0], 
                current_region[-1],
                avg_likelihood
            ))
    
    # Convert frames to time
    result = []
    for start_frame, end_frame, confidence in clapping_sections:
        start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
        end_time = librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length)
        result.append((
            chunk_start_time + start_time,
            chunk_start_time + end_time,
            confidence
        ))
    
    return result

def detect_speech(y, sr, chunk_start_time, silence_threshold_db, min_duration_sec=1.0):
    """
    Detect sections that likely contain human speech/conversation.
    
    Speech typically has:
    1. Moderate spectral centroid (lower than clapping, higher than bass)
    2. Strong spectral rolloff in the voice frequency range 
    3. Higher MFCC variance than music in certain coefficients
    4. Fluctuating energy levels
    
    Parameters:
    -----------
    y : np.ndarray
        Audio signal
    sr : int
        Sample rate
    chunk_start_time : float
        Start time of this chunk in the overall recording
    silence_threshold_db : float
        Threshold below which audio is considered silence
    min_duration_sec : float
        Minimum duration for a speech section
        
    Returns:
    --------
    list
        List of (start_time, end_time, confidence) tuples for detected speech sections
    """
    if len(y) < sr * 3:  # Need at least 3 seconds
        return []
        
    hop_length = 512
    frame_length = 2048
    
    # === 1. Extract speech-relevant features ===
    
    # RMS energy and fluctuation
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = librosa.amplitude_to_db(rms, ref=np.max)
    
    # Energy fluctuation - speech has characteristic syllabic patterns (4-8 Hz fluctuations)
    # Calculate energy fluctuation in the 4-8 Hz range (syllabic rate)
    if len(rms) > sr/(2*hop_length):
        # Compute energy envelope fluctuation
        # Get spectrum of energy envelope
        rms_spec = np.abs(np.fft.rfft(rms - np.mean(rms)))
        # Get corresponding frequencies
        rms_freqs = np.fft.rfftfreq(len(rms), 1.0/(sr/hop_length))
        # Find energy in syllabic range (4-8 Hz)
        syllabic_range = (rms_freqs >= 4) & (rms_freqs <= 8)
        syllabic_energy = np.sum(rms_spec[syllabic_range])
        # Normalize by total energy
        total_energy = np.sum(rms_spec) + 1e-8
        syllabic_ratio = syllabic_energy / total_energy
    else:
        syllabic_ratio = 0
    
    # Spectral centroid (speech is mid-range: ~1000-3000 Hz)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length, roll_percent=0.85)[0]
    
    # MFCCs - particularly useful for speech detection
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    
    # Calculate MFCC dynamics (variance over time) - speech has more dynamic MFCCs
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_var = np.var(mfccs, axis=1)
    mfcc_delta_var = np.var(mfcc_delta, axis=1)
    
    # Voice-focused MFCC variance (coefficients 1-6 are more speech-relevant)
    speech_mfcc_var = np.mean(mfcc_var[1:7])
    speech_mfcc_delta_var = np.mean(mfcc_delta_var[1:7])
    
    # Spectral flatness - speech is less flat than noise, more flat than tonal music
    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
    
    # === 2. Compute frequency band energy ratios ===
    
    # Get the STFT 
    D = librosa.stft(y, n_fft=frame_length, hop_length=hop_length)
    
    # Compute power spectrogram
    S = np.abs(D)**2
    
    # Define frequency band ranges (in Hz)
    # Speech fundamental typically occupies 85-255 Hz (male) and 165-255 Hz (female)
    # Formants occupy broader ranges up to ~5000 Hz
    freq_bins = librosa.core.fft_frequencies(sr=sr, n_fft=frame_length)
    
    speech_low_idx = (freq_bins >= 80) & (freq_bins <= 260)  # Fundamental frequency
    speech_mid_idx = (freq_bins >= 260) & (freq_bins <= 2000)  # First formants
    speech_high_idx = (freq_bins >= 2000) & (freq_bins <= 5000)  # Higher formants
    
    # Sum energy in each band for each frame
    speech_low_energy = np.sum(S[speech_low_idx, :], axis=0)
    speech_mid_energy = np.sum(S[speech_mid_idx, :], axis=0)
    speech_high_energy = np.sum(S[speech_high_idx, :], axis=0)
    
    # Compute formant ratios typical for speech
    formant_ratio1 = speech_mid_energy / (speech_low_energy + 1e-8)
    formant_ratio2 = speech_high_energy / (speech_mid_energy + 1e-8)
    
    # Normalize these ratios
    formant_ratio1_norm = (formant_ratio1 - np.min(formant_ratio1)) / (np.max(formant_ratio1) - np.min(formant_ratio1) + 1e-8)
    formant_ratio2_norm = (formant_ratio2 - np.min(formant_ratio2)) / (np.max(formant_ratio2) - np.min(formant_ratio2) + 1e-8)
    
    # === 3. Frame-level speech likelihood calculation ===
    
    # Calculate speech likelihood for each frame
    speech_likelihood = np.zeros_like(rms_db)
    
    # Only consider frames with energy above silence
    valid_frames = rms_db > silence_threshold_db
    
    # Speech typically has centroid in specific range (normalize to be highest around typical speech centroids)
    typical_speech_centroid_min = 1000  # Hz
    typical_speech_centroid_max = 3000  # Hz
    
    # Create a weighting function that peaks at speech frequencies
    centroid_weight = np.zeros_like(centroid)
    speech_range = (centroid >= typical_speech_centroid_min) & (centroid <= typical_speech_centroid_max)
    
    # Create bell curve around typical speech range
    if np.any(speech_range):
        # Scale to 0-1, with 1 at midpoint of speech range
        speech_mid = (typical_speech_centroid_min + typical_speech_centroid_max) / 2
        centroid_weight[speech_range] = 1.0 - np.abs(centroid[speech_range] - speech_mid) / (speech_mid - typical_speech_centroid_min)
    
    # Calculate frame-by-frame speech likelihood
    if np.any(valid_frames):
        # Normalize features for valid frames
        flatness_norm = (flatness - np.min(flatness)) / (np.max(flatness) - np.min(flatness) + 1e-8)
        centroid_weight_norm = centroid_weight / np.max(centroid_weight + 1e-8)
        
        # Combine all features
        speech_likelihood[valid_frames] = (
            0.25 * centroid_weight_norm[valid_frames] +  # Weight by distance to typical speech centroid
            0.25 * formant_ratio1_norm[valid_frames] +   # First formant ratio
            0.20 * formant_ratio2_norm[valid_frames] +   # Second formant ratio
            0.15 * flatness_norm[valid_frames] +         # Spectral flatness
            0.15                                         # Base likelihood for valid frames
        )
        
        # Boost by syllabic rhythm if present
        speech_likelihood[valid_frames] *= (1.0 + syllabic_ratio)
        
        # Apply MFCC variance as a global scaling factor
        # Higher variance in certain MFCCs is characteristic of speech
        mfcc_speech_factor = min(speech_mfcc_var * 5.0, 2.0)  # Cap at doubling the likelihood
        speech_likelihood[valid_frames] *= mfcc_speech_factor
        
        # Normalize to 0-1
        if np.max(speech_likelihood) > 0:
            speech_likelihood = np.clip(speech_likelihood / np.max(speech_likelihood), 0, 1)
    
    # === 4. Find continuous sections of speech ===
    
    # Apply threshold
    speech_threshold = 0.6
    potential_speech_frames = np.where(speech_likelihood > speech_threshold)[0]
    
    # Group consecutive frames
    speech_sections = []
    min_frames = int(min_duration_sec * sr / hop_length)
    
    if len(potential_speech_frames) > 0:
        # Group consecutive indices
        region_start = potential_speech_frames[0]
        current_region = [region_start]
        
        for i in range(1, len(potential_speech_frames)):
            # Allow small gaps (up to 0.5 second) to account for pauses in speech
            max_gap = int(0.5 * sr / hop_length)
            if potential_speech_frames[i] <= potential_speech_frames[i-1] + max_gap:
                current_region.append(potential_speech_frames[i])
            else:
                if len(current_region) >= min_frames:
                    avg_likelihood = np.mean(speech_likelihood[current_region])
                    speech_sections.append((
                        current_region[0], 
                        current_region[-1],
                        avg_likelihood
                    ))
                region_start = potential_speech_frames[i]
                current_region = [region_start]
        
        # Don't forget the last region
        if len(current_region) >= min_frames:
            avg_likelihood = np.mean(speech_likelihood[current_region])
            speech_sections.append((
                current_region[0], 
                current_region[-1],
                avg_likelihood
            ))
    
    # Convert frames to time
    result = []
    for start_frame, end_frame, confidence in speech_sections:
        start_time = librosa.frames_to_time(start_frame, sr=sr, hop_length=hop_length)
        end_time = librosa.frames_to_time(end_frame, sr=sr, hop_length=hop_length)
        result.append((
            chunk_start_time + start_time,
            chunk_start_time + end_time,
            confidence
        ))
    
    return result

def identify_song_boundaries(events, proximity_threshold_sec=30):
    """
    Identify high-confidence song boundaries by analyzing the proximity of different indicators.
    
    The following patterns strongly suggest song boundaries:
    1. Silence + Speech/Conversation nearby (typical end-of-song -> talking -> new song)
    2. Clapping + Silence and/or Speech nearby (applause -> talking/silence -> new song)
    3. Significant Tempo change following Silence or Speech
    
    Parameters:
    -----------
    events : list
        List of detected events with timestamps and types
    proximity_threshold_sec : float
        Maximum time difference between events to consider them related/proximate
        
    Returns:
    --------
    list
        List of high-confidence song boundary events with timestamps
    """
    if not events:
        return []
    
    # Sort events by time
    sorted_events = sorted(events, key=lambda x: x["time"])
    
    # Prepare storage for high-confidence boundaries
    song_boundaries = []
    
    # Group events that are close to each other
    event_clusters = []
    current_cluster = [sorted_events[0]]
    
    for i in range(1, len(sorted_events)):
        current_event = sorted_events[i]
        prev_event = sorted_events[i-1]
        
        # Check if this event is close to the previous one
        if current_event["time"] - prev_event["time"] <= proximity_threshold_sec:
            # Add to current cluster
            current_cluster.append(current_event)
        else:
            # Start a new cluster
            if current_cluster:
                event_clusters.append(current_cluster)
            current_cluster = [current_event]
    
    # Don't forget the last cluster
    if current_cluster:
        event_clusters.append(current_cluster)
    
    # Analyze each cluster for song boundary patterns
    for cluster in event_clusters:
        # Extract event types in this cluster
        event_types = [e["type"] for e in cluster]
        
        # Set up default confidence
        boundary_confidence = 0.0
        boundary_time = None
        evidence = []
        
        # Check if this cluster contains indicators of a song boundary
        has_silence = "silence_boundary" in event_types
        has_clapping = "clapping" in event_types
        has_speech = "speech" in event_types
        has_tempo_change = "tempo_change" in event_types
        has_structural_boundary = "structural_boundary" in event_types
        
        # Pattern 1: Silence + Speech (high confidence)
        if has_silence and has_speech:
            boundary_confidence = max(boundary_confidence, 0.9)
            evidence.append("silence+speech")
        
        # Pattern 2: Clapping + (Silence or Speech)
        if has_clapping and (has_silence or has_speech):
            boundary_confidence = max(boundary_confidence, 0.85)
            evidence.append("clapping+transition")
        
        # Pattern 3: Tempo change after Silence or Speech
        if has_tempo_change and (has_silence or has_speech):
            # Get the tempo change and silence/speech events
            tempo_events = [e for e in cluster if e["type"] == "tempo_change"]
            transition_events = [e for e in cluster if e["type"] in ["silence_boundary", "speech"]]
            
            # Check if tempo change follows a silence or speech
            for tempo_event in tempo_events:
                for transition_event in transition_events:
                    # If tempo change is after a transition event within the cluster
                    if tempo_event["time"] > transition_event["time"]:
                        # Higher confidence for larger tempo changes
                        change_magnitude = abs(tempo_event["bpm"] - tempo_event["prev_bpm"])
                        tempo_confidence = min(0.5 + change_magnitude/20.0, 0.9)  # Cap at 0.9
                        boundary_confidence = max(boundary_confidence, tempo_confidence)
                        evidence.append(f"tempo_change_{change_magnitude:.1f}bpm")
        
        # Pattern 4: Structural boundary + (Silence or Speech or Clapping)
        if has_structural_boundary and (has_silence or has_speech or has_clapping):
            boundary_confidence = max(boundary_confidence, 0.8)
            evidence.append("structural+transition")
        
        # Pattern 5: All three major indicators (silence, speech, clapping)
        if has_silence and has_speech and has_clapping:
            boundary_confidence = max(boundary_confidence, 0.95)  # Highest confidence
            evidence.append("all_indicators")
            
        # If we have sufficient confidence, create a song boundary
        if boundary_confidence >= 0.7:
            # Find the most appropriate timestamp for the boundary
            # Prefer silence start times
            silence_events = [e for e in cluster if e["type"] == "silence_boundary"]
            if silence_events:
                # Use the first silence in the cluster as the boundary point
                boundary_time = silence_events[0]["time"]
            else:
                # Otherwise use the first event in the cluster
                boundary_time = cluster[0]["time"]
            
            # Create boundary event
            song_boundaries.append({
                "time": boundary_time,
                "type": "song_boundary",
                "confidence": boundary_confidence,
                "evidence": "+".join(evidence)
            })
    
    # Remove duplicate boundaries (those that are too close to each other)
    if song_boundaries:
        song_boundaries.sort(key=lambda x: x["time"])
        deduplicated_boundaries = [song_boundaries[0]]
        
        for i in range(1, len(song_boundaries)):
            current = song_boundaries[i]
            previous = deduplicated_boundaries[-1]
            
            # If this boundary is far enough from the previous one
            if current["time"] - previous["time"] > proximity_threshold_sec:
                deduplicated_boundaries.append(current)
            else:
                # Keep the higher confidence boundary
                if current["confidence"] > previous["confidence"]:
                    deduplicated_boundaries[-1] = current
        
        song_boundaries = deduplicated_boundaries
    
    return song_boundaries

def main():
    parser = argparse.ArgumentParser(description='Analyze a long jam session recording for tempo changes and song boundaries.')
    parser.add_argument('input_file', help='Path to the audio file to analyze')
    parser.add_argument('--output', '-o', help='Output JSON file for results (optional)')
    parser.add_argument('--chunk-duration', type=float, default=300, help='Duration of processing chunks in seconds (default: 300)')
    parser.add_argument('--overlap', type=float, default=5, help='Overlap between chunks in seconds (default: 5)')
    parser.add_argument('--tempo-threshold', type=float, default=5.0, help='Minimum BPM difference for tempo change detection (default: 5)')
    parser.add_argument('--boundary-threshold', type=float, default=0.15, help='Threshold for boundary detection (default: 0.15)')
    parser.add_argument('--silence-threshold', type=float, default=-50, help='Silence threshold in dB (default: -50)')
    parser.add_argument('--silence-duration', type=float, default=2.0, help='Minimum silence duration in seconds (default: 2)')
    parser.add_argument('--min-segment', type=float, default=30, help='Minimum segment length in seconds (default: 30)')
    parser.add_argument('--clapping-duration', type=float, default=1.0, help='Minimum clapping duration in seconds (default: 1)')
    parser.add_argument('--clapping-percentile', type=int, default=20, help='Energy percentile for clapping detection (default: 20)')
    parser.add_argument('--clapping-irregularity', type=float, default=0.6, help='Irregularity threshold for clapping (default: 0.6)')
    parser.add_argument('--speech-duration', type=float, default=1.0, help='Minimum speech duration in seconds (default: 1.0)')
    parser.add_argument('--proximity-threshold', type=float, default=30, help='Maximum time between related events for song boundary detection (default: 30)')
    parser.add_argument('--no-song-boundaries', action='store_true', help='Disable high-confidence song boundary identification')
    parser.add_argument('--quiet', '-q', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    try:
        results = detect_events(
            args.input_file,
            chunk_duration_sec=args.chunk_duration,
            overlap_sec=args.overlap,
            tempo_change_threshold_bpm=args.tempo_threshold,
            boundary_peak_threshold=args.boundary_threshold,
            silence_threshold_db=args.silence_threshold,
            silence_min_duration_sec=args.silence_duration,
            min_segment_length_sec=args.min_segment,
            clapping_min_duration_sec=args.clapping_duration,
            clapping_energy_percentile=args.clapping_percentile,
            clapping_irregularity_threshold=args.clapping_irregularity,
            speech_min_duration_sec=args.speech_duration,
            identify_song_boundaries_flag=not args.no_song_boundaries,
            proximity_threshold_sec=args.proximity_threshold,
            verbose=not args.quiet
        )
        
        elapsed = time.time() - start_time
        
        # Print summary
        if not args.quiet:
            print("\n" + "=" * 60)
            print(f"Analysis complete! Found {len(results)} events in {elapsed:.1f} seconds")
            print("=" * 60)
            print("\nTIMESTAMP  | TYPE                | DETAILS")
            print("-" * 60)
            
            for event in results:
                # Skip speech events in the output
                if event["type"] == "speech":
                    continue
                    
                timestamp = format_timestamp(event["time"])
                event_type = event["type"].replace("_", " ").title()
                
                details = ""
                if event["type"] == "tempo_change":
                    details = f"{event['prev_bpm']:.1f} → {event['bpm']:.1f} BPM"
                elif event["type"] == "silence_boundary":
                    details = f"Silence: {event['duration']:.1f}s"
                elif event["type"] == "clapping":
                    end_timestamp = format_timestamp(event["end_time"])
                    details = f"Duration: {event['duration']:.1f}s, Confidence: {event['confidence']:.2f}"
                    timestamp = f"{timestamp} - {end_timestamp}"
                elif event["type"] == "song_boundary":
                    details = f"Confidence: {event['confidence']:.2f}, Evidence: {event['evidence']}"
                    
                print(f"{timestamp} | {event_type:20} | {details}")
        
        # Save results if output file specified
        if args.output:
            with open(args.output, 'w') as f:
                # Add metadata
                output_data = {
                    "file": args.input_file,
                    "duration": file_duration_sec if 'file_duration_sec' in locals() else None,
                    "analysis_time": elapsed,
                    "parameters": {
                        "chunk_duration": args.chunk_duration,
                        "overlap": args.overlap,
                        "tempo_threshold": args.tempo_threshold,
                        "boundary_threshold": args.boundary_threshold,
                        "silence_threshold": args.silence_threshold,
                        "silence_duration": args.silence_duration,
                        "min_segment": args.min_segment,
                        "clapping_duration": args.clapping_duration,
                        "clapping_percentile": args.clapping_percentile,
                        "clapping_irregularity": args.clapping_irregularity,
                        "speech_duration": args.speech_duration,
                        "proximity_threshold": args.proximity_threshold,
                        "identify_song_boundaries": not args.no_song_boundaries
                    },
                    # Filter out speech events from the output
                    "events": [e for e in results if e["type"] != "speech"]
                }
                json.dump(output_data, f, indent=2)
                
            if not args.quiet:
                print(f"\nResults saved to {args.output}")
                
                # Print song boundary summary if there are any
                song_boundaries = [e for e in results if e["type"] == "song_boundary"]
                if song_boundaries:
                    print("\n" + "=" * 60)
                    print(f"SONG BOUNDARIES SUMMARY: {len(song_boundaries)} identified")
                    print("=" * 60)
                    
                    for i, boundary in enumerate(song_boundaries):
                        timestamp = format_timestamp(boundary["time"])
                        print(f"{i+1}. {timestamp} (Confidence: {boundary['confidence']:.2f})")
                        
                        # Calculate song duration if we have more than one boundary
                        if i > 0:
                            prev_time = song_boundaries[i-1]["time"]
                            duration = boundary["time"] - prev_time
                            print(f"   Song duration: {format_timestamp(duration)}")
                
    except Exception as e:
        print(f"Error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main()) 