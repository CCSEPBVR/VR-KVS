/*****************************************************************************/
/**
 *  @file   FrameEvent.cpp
 */
/*----------------------------------------------------------------------------
 *
 *  Copyright (c) Visualization Laboratory, Kyoto University.
 *  All rights reserved.
 *  See http://www.viz.media.kyoto-u.ac.jp/kvs/copyright/ for details.
 *
 *  $Id$
 */
/*****************************************************************************/
#include "FrameEvent.h"

namespace kvs
{

/*===========================================================================*/
/**
 *  @brief  Constructs a new FrameEvent class.
 */
/*===========================================================================*/
FrameEvent::FrameEvent()
{
}

/*===========================================================================*/
/**
 *  @brief  Destroys the FrameEvent class.
 */
/*===========================================================================*/
FrameEvent::~FrameEvent()
{
}

/*===========================================================================*/
/**
 *  @brief  Returns the event type.
 *  @return event type
 */
/*===========================================================================*/
int FrameEvent::type() const
{
    return EventBase::FrameEvent;
}

} // end of namespace kvs
