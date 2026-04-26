(region DMN
  (def layers 3)
  (def connections 'skip)
  (def learning_rule 'backprop)
  (def novelty_threshold 0.3)
  (def thought_alpha 0.7)

  ;; Projections: (projection src tgt type nt condition)
  (def projections
    [ (projection 'DMN 'PFC 'signal nil (lambda (ctx) (> (get ctx 'novelty) novelty_threshold)))
      (projection 'DMN 'Hippocampus 'signal nil (lambda (ctx) true))
      (projection 'DMN 'BG 'nt '5HT (lambda (ctx) (> (get ctx 'stress) 0.5)))
    ])

  ;; NT production: (nt_production nt amount_fn context_fn)
  (def nt_production
    [ (nt_prod '5HT (lambda (ctx) (* 0.6 (get ctx 'thought_alpha))) (lambda (ctx) true))
      (nt_prod 'ACh (lambda (ctx) 0.2) (lambda (ctx) (> (get ctx 'attention) 0.5)))
    ])

  (defun select-candidate (candidates)
    (let ((best nil) (score -inf))
      (foreach c candidates
        (let ((s (score-candidate c novelty_threshold thought_alpha)))
          (if (> s score)
              (set! best c)
              (set! score s))))
      best))

  (defun on-epigenetic-signal (nt-state)
    (if (> (get nt-state 'DA) 0.8)
  (set! novelty_threshold (+ novelty_threshold 0.05))))

)
