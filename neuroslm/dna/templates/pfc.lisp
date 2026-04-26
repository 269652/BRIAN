(region PFC
  (def working_mem_size 8)
  (def layers 4) ; number of transformer layers
  (def connections 'sequential) ; could be 'sequential, 'skip, etc.
  (def learning_rule 'backprop) ; could be 'backprop, 'hebbian, etc.
  (defun update-working-mem (mem input)
    (let ((new-mem (append (list input) (take mem (- working_mem_size 1)))))
      new-mem))
  (defun on-epigenetic-signal (nt-state)
    (if (> (get nt-state 'DA) 0.6)
  (set! working_mem_size (+ working_mem_size 1))))

)
