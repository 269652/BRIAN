
(region BasalGanglia
  (def layers 2)
  (def connections 'sequential)
  (def learning_rule 'reinforce)
  (def action_threshold 0.5)

  ;; Projections
  (def projections
    [ (projection 'BG 'PFC 'signal nil (lambda (ctx) (> (get ctx 'action_prob') 0.5)))
      (projection 'BG 'DMN 'nt 'DA (lambda (ctx) (> (get ctx 'reward') 0.5)))
      (projection 'BG 'Hippocampus 'signal nil (lambda (ctx) true))
    ])

  ;; NT production
  (def nt_production
    [ (nt_prod 'DA (lambda (ctx) (* 0.7 (get ctx 'action_threshold'))) (lambda (ctx) true))
    ])

  (defun select-action (actions)
    (let ((best nil) (score -inf))
      (foreach a actions
        (let ((s (score-action a action_threshold)))
          (if (> s score)
              (set! best a)
              (set! score s))))
      best))

  (defun on-epigenetic-signal (nt-state)
    (if (> (get nt-state 'DA) 0.7)
        (set! action_threshold (- action_threshold 0.05)))))
