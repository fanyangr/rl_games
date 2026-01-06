import torch
import torch.nn as nn
from rl_games.algos_torch import network_builder
from rl_games.algos_torch import torch_ext
from rl_games.common import object_factory

class MultiModalA2CBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            # Bypass A2CBuilder.Network.__init__ to avoid error on input_shape parsing
            # We call NetworkBuilder.BaseNetwork.__init__ directly
            actions_num = kwargs.pop('actions_num')
            input_shape = kwargs.pop('input_shape')
            self.value_size = kwargs.pop('value_size', 1)
            self.num_seqs = num_seqs = kwargs.pop('num_seqs', 1)

            network_builder.NetworkBuilder.BaseNetwork.__init__(self)
            self.load(params)
            
            self.actor_cnn = nn.Sequential()
            self.actor_mlp = nn.Sequential()

            # Handle dictionary input shape
            if isinstance(input_shape, dict):
                # Expect 'visual' and 'vector_obs' - general naming for multi-modal
                # But to support legacy/specific 'tactile' config, we can check for that too or assume 'visual' is the image part
                
                # Logic: Look for image-like keys.
                self.visual_shape = None
                self.vector_shape = None
                
                # Check for known image keys or defaults
                # Priority: 'visual', 'tactile', 'image'
                for key in ['visual', 'tactile', 'image']:
                    if key in input_shape:
                        self.visual_shape = input_shape[key]
                        break
                
                if 'vector_obs' in input_shape:
                    self.vector_shape = input_shape['vector_obs']
                
                if self.visual_shape is None and self.vector_shape is None:
                     # Fallback to handling it as standard if keys are missing
                     print(f"MultiModalA2CBuilder: Warning - input_shape is dict but missing expected keys (visual/tactile/image, vector_obs). Keys: {input_shape.keys()}")
                
                print(f"MultiModalA2CBuilder: Visual shape: {self.visual_shape}, Vector shape: {self.vector_shape}")
            else:
                # If not dict, behave like standard A2C
                self.visual_shape = None
                self.vector_shape = input_shape
                print(f"MultiModalA2CBuilder: Warning - input_shape is not dict: {input_shape}")

            if self.has_cnn and self.visual_shape is not None:
                # Build CNN for visual input
                cnn_input_shape = self.visual_shape
                if self.permute_input:
                    cnn_input_shape = torch_ext.shape_whc_to_cwh(cnn_input_shape)
                
                cnn_args = {
                    'ctype': self.cnn['type'],
                    'input_shape': cnn_input_shape,
                    'convs': self.cnn['convs'],
                    'activation': self.cnn['activation'],
                    'norm_func_name': self.normalization,
                }
                self.actor_cnn = self._build_conv(**cnn_args)

                # Calculate CNN output size
                cnn_output_size = self._calc_input_size(cnn_input_shape, self.actor_cnn)
            else:
                cnn_output_size = 0
            
            # Calculate MLP input size
            if self.vector_shape is not None:
                # If vector_shape is tuple (N,), take N.
                if hasattr(self.vector_shape, '__len__'):
                    vector_dim = self.vector_shape[0]
                else:
                    vector_dim = self.vector_shape
            else:
                 vector_dim = 0

            mlp_input_size = cnn_output_size + vector_dim
            print(f"MultiModalA2CBuilder: MLP input size: {mlp_input_size} (CNN: {cnn_output_size} + Vec: {vector_dim})")


            # Build MLP
            if len(self.units) == 0:
                out_size = mlp_input_size
            else:
                out_size = self.units[-1]

            # RNN handling (Simplified - assume no RNN or standard RNN on combined features)
            if self.has_rnn:
                # Logic copied/adapted from A2CBuilder
                if not self.is_rnn_before_mlp:
                    rnn_in_size = out_size
                    # concat logic omitted for brevity unless needed
                    out_size = self.rnn_units
                else:
                    rnn_in_size = mlp_input_size
                    mlp_input_size = self.rnn_units
                
                self.rnn = self._build_rnn(self.rnn_name, rnn_in_size, self.rnn_units, self.rnn_layers)
                if self.rnn_ln:
                    self.layer_norm = torch.nn.LayerNorm(self.rnn_units)

            mlp_args = {
                'input_size': mlp_input_size,
                'units': self.units,
                'activation': self.activation,
                'norm_func_name': self.normalization,
                'dense_func': torch.nn.Linear,
                'd2rl': self.is_d2rl,
                'norm_only_first_layer': self.norm_only_first_layer
            }
            self.actor_mlp = self._build_mlp(**mlp_args)

            # Value function (Standard)
            self.value = self._build_value_layer(out_size, self.value_size)
            self.value_act = self.activations_factory.create(self.value_activation)

            # Heads
            if self.is_discrete:
                self.logits = torch.nn.Linear(out_size, actions_num)
            if self.is_multi_discrete:
                self.logits = torch.nn.ModuleList([torch.nn.Linear(out_size, num) for num in actions_num])
            if self.is_continuous:
                self.mu = torch.nn.Linear(out_size, actions_num)
                self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
                mu_init = self.init_factory.create(**self.space_config['mu_init'])
                self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
                sigma_init = self.init_factory.create(**self.space_config['sigma_init'])

                if self.fixed_sigma:
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                else:
                    self.sigma = torch.nn.Linear(out_size, actions_num)

            # Initialization
            mlp_init = self.init_factory.create(**self.initializer)
            if self.has_cnn:
                cnn_init = self.init_factory.create(**self.cnn['initializer'])

            for m in self.modules():         
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                    cnn_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias)

            if self.is_continuous:
                mu_init(self.mu.weight)
                if self.fixed_sigma:
                    sigma_init(self.sigma)
                else:
                    sigma_init(self.sigma.weight)


        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0)

            # Split Observation
            if isinstance(obs, dict):
                # Try to get visual input
                visual = None
                for key in ['visual', 'tactile', 'image']:
                    if key in obs:
                        visual = obs[key]
                        break
                
                vector = obs.get('vector_obs')
            else:
                # Fallback if somehow not dict
                visual = None
                vector = obs

            # Process Visual
            if self.has_cnn and visual is not None:
                # Permute if needed (N, H, W, C) -> (N, C, H, W)
                if self.permute_input and len(visual.shape) == 4:
                    visual = visual.permute((0, 3, 1, 2))
                
                cnn_out = self.actor_cnn(visual)
                cnn_out = cnn_out.flatten(1)
            else:
                cnn_out = None

            # Concatenate
            if cnn_out is not None and vector is not None:
                mlp_in = torch.cat([vector, cnn_out], dim=1)
            elif cnn_out is not None:
                mlp_in = cnn_out
            else:
                mlp_in = vector

            out = mlp_in
            
            # MLP Pass
            if self.has_rnn:
                # Simplified RNN logic (assuming single stream)
                # ... (omitted complex separate/concat rnn logic for brevity, assume standard flow)
                
                if not self.is_rnn_before_mlp:
                    out = self.actor_mlp(out)

                batch_size = out.size()[0]
                seq_length = obs_dict.get('seq_length', 1)
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1)
                
                if states is None:
                     states = self.get_default_rnn_state()
                     # Expand to batch? usually rl_games handles this or we just pass defaults
                     # Wait, states are passed in from runner.

                if len(states) == 1:
                    states = states[0]

                out = out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1)
                    dones = dones.transpose(0, 1)
                
                out, states = self.rnn(out, states, dones, bptt_len)
                out = out.transpose(0, 1)
                out = out.contiguous().reshape(out.size()[0] * out.size()[1], -1)

                if self.rnn_ln:
                    out = self.layer_norm(out)
                
                if self.is_rnn_before_mlp:
                    out = self.actor_mlp(out)
                
                if type(states) is not tuple:
                    states = (states,)
            else:
                out = self.actor_mlp(out)

            # Value Head
            value = self.value_act(self.value(out))

            # Action Heads
            if self.is_discrete:
                logits = self.logits(out)
                return logits, value, states
            
            if self.is_multi_discrete:
                logits = [logit(out) for logit in self.logits]
                return logits, value, states
            
            if self.is_continuous:
                mu = self.mu_act(self.mu(out))
                if self.fixed_sigma:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(out))
                
                return mu, sigma, value, states

    def build(self, name, **kwargs):
        net = MultiModalA2CBuilder.Network(self.params, **kwargs)
        return net

