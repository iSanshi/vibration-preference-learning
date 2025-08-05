import numpy as np

def generate_tone_signal(filler_amplitude, filler_frequency, filler_density, 
                        filler_env_gradient, duration, cycles, fs):
    """
    Generate a tone signal with specified parameters.
    
    Parameters:
    -----------
    filler_amplitude : float
        Amplitude of the filler (range: 0-100)
    filler_frequency : float
        Frequency of the filler (range: 0-100)
    filler_density : float
        Density of the filler (range: 0-100)
    filler_env_gradient : float
        Envelope gradient of the filler (range: -100-100)
    duration : float
        Duration in seconds (range: 1-30)
    cycles : int
        Number of cycles (range: 1-30)
    fs : float
        Sampling frequency
        
    Returns:
    --------
    time : ndarray
        Time vector
    data : ndarray
        Generated signal data
    for_plot : dict
        Dictionary containing additional plotting data
    """
    # Convert inputs to float
    pattern = {
        'filler': {
            'amplitude': float(filler_amplitude),
            'frequency': float(filler_frequency),
            'density': float(filler_density),
            'envelope': {'relative_gradient': float(filler_env_gradient)}
        },
        'duration': float(duration),
        'cycle': float(cycles)
    }
    fs = float(fs)

    # Mapping & EQ
    # Frequency
    absolute_frequency = (250-50)/100 * pattern['filler']['frequency'] + 50

    # Amplitude scaling
    if 110 < absolute_frequency <= 130:
        scale = (0.2-1)/(130-110)*absolute_frequency - (0.2-1)/(130-110)*130 + 0.2
    elif 130 < absolute_frequency < 150:
        scale = (1-0.2)/(150-130)*absolute_frequency - (1-0.2)/(150-130)*130 + 0.2
    else:
        scale = 1
    
    absolute_amplitude = pattern['filler']['amplitude']/100 * scale

    # Generate pattern
    filler_time = np.arange(0, pattern['duration'], 1/fs)
    len_signal = len(filler_time)

    # Density
    pic = 100e-3
    granularity = pic/2 * 5
    nu = round(pattern['duration']/granularity)
    granularity = round(pattern['duration']/nu, 3)
    pic = granularity/5 * 2

    if pattern['filler']['density'] <= 50:
        boundary = -pattern['filler']['density']/50 + 1
        fade_number = round(pic * fs)
        upper = np.linspace(boundary, 1, fade_number)
        downer = np.linspace(1, boundary, fade_number)
    else:
        fade_number = round(((0.1-1)/50 * pattern['filler']['density'] - 
                           (0.1-1)/50 * 50 + 1) * pic * fs)
        upper = np.linspace(0, 1, fade_number)
        before_upper = np.zeros(round(pic*fs - len(upper)))
        upper = np.concatenate([before_upper, upper])
        downer = np.linspace(1, 0, fade_number)
        after_downer = np.zeros(round(pic*fs - len(downer)))
        downer = np.concatenate([downer, after_downer])

    keeper = np.ones(round(granularity*fs) - len(upper) - len(downer))
    env = np.concatenate([upper, keeper, downer])
    num = round(granularity*fs)
    
    # Build envelope
    envelope = np.array([])
    j = 1
    id_val = j * num
    while id_val <= len_signal:
        envelope = np.concatenate([envelope, env])
        j += 1
        id_val = j * num

    if len(envelope) < len_signal:
        envelope = np.concatenate([envelope, env[:len_signal-len(envelope)]])
    elif len(envelope) > len_signal:
        raise ValueError('This should be impossible')

    # Filler relative gradient
    if pattern['filler']['envelope']['relative_gradient'] <= 0:
        p1 = (100 + pattern['filler']['envelope']['relative_gradient'])/100
        p2 = 1
    else:
        p1 = 1
        p2 = (100 - pattern['filler']['envelope']['relative_gradient'])/100
    
    filler_env = np.linspace(p1, p2, len_signal)

    # Fade mask
    fm1 = np.linspace(0, 1, int(10e-3*fs))
    fm2 = np.linspace(1, 0, int(10e-3*fs))
    fm = np.ones(len_signal)
    fm[:len(fm1)] = fm1
    fm[-len(fm2):] = fm2

    # Generate pattern
    base_filler = absolute_amplitude * np.sin(2*np.pi*absolute_frequency*filler_time)
    conv_density = base_filler * envelope
    conv_envelope = conv_density * filler_env
    conv_fade_mask = conv_envelope * fm

    # Combine pattern
    data = np.tile(conv_fade_mask, int(pattern['cycle']))
    time = np.arange(len(data))/fs

    # Round outputs
    data = np.round(data, 5)
    time = np.round(time, 6)

    # Prepare plotting data
    for_plot = {
        'for_density': envelope * absolute_amplitude,
        'for_envelope': filler_env * absolute_amplitude,
        'filler_time': filler_time
    }

    return time, data, for_plot